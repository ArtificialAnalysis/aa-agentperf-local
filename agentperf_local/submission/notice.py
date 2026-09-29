"""State the privacy notice a person accepts before a submission leaves this computer.

Public surface: PRIVACY_NOTICE and PRIVACY_NOTICE_VERSION.
"""

# Every submission names the notice version its sender accepted, so change the version
# whenever the wording changes. The service treats a different version as different content.
PRIVACY_NOTICE_VERSION = "2026-09-29"
PRIVACY_NOTICE = (
    "Artificial Analysis may publish aggregate results and per-turn timings from this run.\n"
    "The service keeps the whole submission in private storage indefinitely. It includes the hardware,\n"
    "the model and framework build, and the server launch command.\n"
    "Prompts, responses, credentials, local paths, hostnames, serial numbers, and endpoint URLs are never sent.\n"
    "Failed qualification probes are recorded, not rejected. A copy stays on this computer;\n"
    "sending it again does not create a second submission."
)
