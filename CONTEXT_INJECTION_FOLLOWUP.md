# Context injection follow-up

The context-injection assignment is already present on `main`. This follow-up
adds `AugmentedPrompt.to_messages()`, which exposes the assembled grounded
prompt in the exact `system` and `user` message shape expected by a chat
completion client.

The original implementation still enforces the token budget before these
messages are returned, preserves `[1]`-style source markers, and instructs the
model to answer only from the supplied context.