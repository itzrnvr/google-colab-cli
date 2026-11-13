# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import jupyter_kernel_client
import requests

# How long to wait for the kernel to acknowledge an interrupt before giving up
# on the worker thread and reporting the timeout.
INTERRUPT_GRACE_SECONDS = 5.0


class ExecutionTimeout(TimeoutError):
    """Raised when a kernel execution exceeds its wall-clock budget.

    This is deliberately *not* the same as the upstream
    ``jupyter_kernel_client`` timeout: that one is an idle budget and, once
    the deadline passes, degenerates into a busy spin instead of raising.
    See ``ColabRuntime._run_with_deadline``.
    """


class ColabRuntime:
    def __init__(
        self,
        url: str,
        token: str,
        session_name: Optional[str] = None,
        history: Optional[Any] = None,
        kernel_id: Optional[str] = None,
        session_id: Optional[str] = None,
        on_kernel_started: Optional[Callable[[str], None]] = None,
        on_session_started: Optional[Callable[[str], None]] = None,
    ):
        self.url = url
        self.token = token
        self.session_name = session_name
        self.history = history
        self.kernel_id = kernel_id
        self.session_id = session_id
        self.on_kernel_started = on_kernel_started
        self.on_session_started = on_session_started
        self._kernel_client = None
        self.colab_request_hook: Optional[Callable[[Dict[str, Any], Any], None]] = None

    def _apply_ws_hook(self):
        wsclient = self._kernel_client._manager.client
        original_on_message = wsclient.kernel_socket.on_message

        def hooked_on_message(s_ws, message):
            if not self.colab_request_hook:
                return original_on_message(s_ws, message)

            try:
                from jupyter_kernel_client.wsclient import JupyterSubprotocol

                if wsclient._subprotocol == JupyterSubprotocol.DEFAULT:
                    from jupyter_kernel_client.wsclient import (
                        deserialize_msg_from_ws_default,
                    )

                    deserialize_msg = deserialize_msg_from_ws_default(message)
                elif wsclient._subprotocol == JupyterSubprotocol.V1:
                    from jupyter_kernel_client.wsclient import (
                        deserialize_msg_from_ws_v1,
                    )

                    channel, msg_list = deserialize_msg_from_ws_v1(message)
                    deserialize_msg = wsclient.session.deserialize(msg_list)
                else:
                    deserialize_msg = None

                if deserialize_msg:
                    msg_type = deserialize_msg.get("msg_type")
                    if msg_type == "colab_request":
                        # We pass the deserialized msg and the wsclient to the hook
                        if self.colab_request_hook(deserialize_msg, wsclient):
                            # If the hook returns True, we intercept and do NOT pass to original
                            return

            except Exception as e:
                logging.debug(f"Error in colab_request hook: {e}")

            # Call original for all other messages
            original_on_message(s_ws, message)

        wsclient.kernel_socket.on_message = hooked_on_message

    @property
    def kernel_client(self):
        if not self._kernel_client:
            retries = 3
            backoff = 2
            last_err = None

            for i in range(retries):
                try:
                    client_kwargs = {
                        "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                        "extra_params": {"colab-runtime-proxy-token": self.token},
                    }
                    if self.session_id:
                        # WSSession (Session) expects 'session' for the ID
                        client_kwargs["session"] = self.session_id

                    # Resolve the kernel client class across
                    # jupyter-kernel-client variants:
                    # - git (googlecolab fork): ColabKernelClient(proxy_token=...)
                    # - PyPI <1.0: KernelClient(token=...)
                    # - PyPI >=1.0: JupyterKernelClient(token=...)
                    #   (KernelClient was renamed, breaking `colab-tool exec` with
                    #   AttributeError when installed from PyPI).
                    if hasattr(jupyter_kernel_client, "ColabKernelClient"):
                        _ClientCls = jupyter_kernel_client.ColabKernelClient
                        _token_kw = "proxy_token"
                    elif hasattr(jupyter_kernel_client, "KernelClient"):
                        _ClientCls = jupyter_kernel_client.KernelClient
                        _token_kw = "token"
                    else:
                        _ClientCls = jupyter_kernel_client.JupyterKernelClient
                        _token_kw = "token"
                    self._kernel_client = _ClientCls(
                        server_url=self.url,
                        **{_token_kw: self.token},
                        kernel_id=self.kernel_id,
                        client_kwargs=client_kwargs,
                        headers={
                            "X-Colab-Client-Agent": "colab-cli",
                            "X-Colab-Runtime-Proxy-Token": self.token,
                        },
                    )
                    # Force _own_kernel to False. This prevents jupyter-kernel-client
                    # from automatically deleting the kernel when the client is closed or deleted.
                    self._kernel_client._own_kernel = False

                    self._kernel_client.start()
                    self._apply_ws_hook()

                    # Capture IDs if we started fresh
                    if not self.kernel_id and self._kernel_client.id:
                        self.kernel_id = self._kernel_client.id
                        if self.on_kernel_started:
                            self.on_kernel_started(self.kernel_id)

                    if (
                        not self.session_id
                        and self._kernel_client._manager.client.session.session
                    ):
                        self.session_id = (
                            self._kernel_client._manager.client.session.session
                        )
                        if self.on_session_started:
                            self.on_session_started(self.session_id)
                    break
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectTimeout,
                ) as e:
                    last_err = e
                    if i < retries - 1:
                        sleep_time = backoff ** (i + 1)
                        logging.debug(
                            f"Kernel startup timeout, retrying in {sleep_time}s..."
                            f" ({i + 1}/{retries})"
                        )
                        time.sleep(sleep_time)
                    else:
                        raise last_err
                except Exception as e:
                    raise e

        return self._kernel_client

    def restart(
        self,
        timeout: Optional[float] = None,
    ):
        self.kernel_client.restart(timeout=timeout)

    def _interrupt_kernel(self) -> None:
        """Best-effort interrupt of the running kernel."""
        client = self._kernel_client
        if client is None:
            return
        try:
            client.interrupt()
        except Exception as e:  # pragma: no cover - best effort
            logging.debug(f"Could not interrupt kernel: {e}")

    def _run_with_deadline(
        self,
        fn: Callable[[], Any],
        timeout: Optional[float],
        description: str,
        last_activity: Optional[Callable[[], float]] = None,
        max_time: Optional[float] = None,
    ) -> Any:
        """Run ``fn`` under a deadline, enforcing it from the outside.

        ``jupyter_kernel_client`` cannot be trusted with the deadline:

        * With a ``timeout`` it recomputes ``timeout = max(0, deadline - now)``
          and then loops with ``wait(0)`` / ``get_msg(timeout=0)``. ``wait(0)``
          returns immediately and ``get_msg(timeout=0)`` raises ``Empty``, so
          the poll loop spins at 100% CPU and **never** raises
          ``TimeoutError``. The remote cell keeps running (still consuming
          Colab compute quota) and the CLI never returns.
        * Without a ``timeout`` it uses its own 10 s default, which has the
          same fate.

        So the kernel call is always made with ``timeout=None`` (block until
          a message arrives, no spin) and the budget is enforced here.

        ``timeout`` semantics:

        * With ``last_activity`` (the streaming path) it is an **idle
          limit**: the number of seconds allowed between two messages from
          the execution. A run that keeps printing can take as long as it
          needs -- which is what a training job does.
        * Without it (the non-streaming path) it is an absolute wall-clock
          limit.

        ``None`` or a value <= 0 means "no limit" unless ``max_time`` is set.
        ``max_time`` is always an absolute cap.

        On expiry the kernel is interrupted -- which makes the spinning poll
        loop observe the reply and unwind -- and ``ExecutionTimeout`` is
        raised.
        """
        idle_limit = None
        absolute_limit = max_time

        if timeout is not None and timeout > 0:
            if last_activity is not None:
                idle_limit = timeout
            else:
                absolute_limit = timeout if absolute_limit is None else absolute_limit

        if idle_limit is None and absolute_limit is None:
            return fn()

        result: Dict[str, Any] = {}

        def worker() -> None:
            try:
                result["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                result["error"] = exc

        thread = threading.Thread(target=worker, name="colab-exec", daemon=True)
        thread.start()

        started = time.monotonic()
        while thread.is_alive():
            now = time.monotonic()
            limits = []
            if idle_limit is not None:
                limits.append(idle_limit - (now - last_activity()))
            if absolute_limit is not None:
                limits.append(absolute_limit - (now - started))
            if limits and min(limits) <= 0:
                break
            # join() returns as soon as the worker finishes, so this waits
            # for either completion or the next limit.
            thread.join(min(limits) if limits else 0.5)

        if thread.is_alive():
            self._interrupt_kernel()
            thread.join(INTERRUPT_GRACE_SECONDS)
            if thread.is_alive():
                if idle_limit is not None:
                    what = (
                        f"produced no output for {idle_limit:g}s "
                        f"(the --timeout idle limit)"
                    )
                else:
                    what = f"exceeded the {absolute_limit:g}s timeout"
                raise ExecutionTimeout(
                    f"{description} {what} and the kernel did not acknowledge "
                    "the interrupt. The remote cell may still be running and "
                    "will keep consuming compute quota - stop the session "
                    "(`colab-tool stop -s <name>`) to release it."
                )

        if "error" in result:
            raise result["error"]
        return result.get("value")

    def execute_code(
        self,
        code: str,
        allow_stdin: bool = False,
        stdin_hook: Any = None,
        output_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        # The kernel call is always made with ``timeout=None``: with any
        # timeout the client turns the deadline into a 100% CPU spin that
        # never raises, and with its own 10 s default it does the same after
        # 10 s. The budget is enforced by ``_run_with_deadline`` instead --
        # as an idle limit when output is streamed, an absolute limit
        # otherwise.
        kwargs = {"allow_stdin": allow_stdin, "timeout": None}

        # Wrap stdin_hook to log inputs
        original_stdin_hook = stdin_hook

        def wrapped_stdin_hook(prompt):
            if self.history and self.session_name:
                self.history.log_event(
                    self.session_name, "stdin_request", {"prompt": prompt}
                )

            res = original_stdin_hook(prompt) if original_stdin_hook else input(prompt)

            if self.history and self.session_name:
                self.history.log_event(self.session_name, "input_reply", {"value": res})
            return res

        if allow_stdin:
            kwargs["stdin_hook"] = wrapped_stdin_hook

        if output_hook:
            # If we have an output hook, we use execute_interactive and manage buffering ourselves
            outputs = []

            # The idle clock: every message from the kernel resets it.
            last_message = [time.monotonic()]

            def wrapped_output_hook(msg):
                from jupyter_kernel_client.client import (
                    output_hook as default_output_hook,
                )

                last_message[0] = time.monotonic()

                # Update local outputs list using the default logic
                new_indexes = default_output_hook(outputs, msg)
                # If new outputs were added, call our streaming hook with the new data
                if new_indexes:
                    for idx in sorted(new_indexes):
                        if idx < len(outputs):
                            output_hook(outputs[idx])

            reply = self._run_with_deadline(
                lambda: self.kernel_client.execute_interactive(
                    code, output_hook=wrapped_output_hook, **kwargs
                ),
                timeout,
                "Code execution",
                last_activity=lambda: last_message[0],
            )
            # execute_interactive returns the raw reply message
            reply_content = reply["content"] if reply else {"status": "error"}
        else:
            reply = self._run_with_deadline(
                lambda: self.kernel_client.execute(code, **kwargs),
                timeout,
                "Code execution",
            )
            if not reply:
                return []
            outputs = reply.get("outputs", [])
            reply_content = reply

        # If there's an error status but no error in outputs, synthesize one
        if reply_content.get("status") == "error":
            has_error_output = any(o.get("output_type") == "error" for o in outputs)
            if not has_error_output:
                outputs.append(
                    {
                        "output_type": "error",
                        "ename": reply_content.get("ename", "Error"),
                        "evalue": reply_content.get("evalue", "Unknown error"),
                        "traceback": reply_content.get("traceback", []),
                    }
                )

        return outputs

    def stop(self, shutdown_kernel: bool = False):
        if self._kernel_client:
            try:
                # We manage kernel lifecycle explicitly.
                # To prevent automatic shutdown, we bypass the manager's stop() and
                # directly close the channels and socket.
                client = self._kernel_client._manager.client
                client.stop_channels()
                if client.kernel_socket:
                    client.kernel_socket.close()

                if shutdown_kernel:
                    self._kernel_client._manager.shutdown_kernel(now=True)
            except Exception:
                logging.exception("Error stopping kernel client")
