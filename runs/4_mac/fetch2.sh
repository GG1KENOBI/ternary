#!/bin/bash
cd ~/tern-mac/models; H=${SERVER:?укажите SERVER=user@host}
for f in s1/mac/e2-TQ2_0.gguf s1/mac/base08-Q4_K_M.gguf s1/mac/base08-Q8_0.gguf; do scp -q "$H:$f" . && echo "ok $f"; done; echo DONE
