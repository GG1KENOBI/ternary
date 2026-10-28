#!/bin/bash
cd ~/tern-mac/models
H=${SERVER:?укажите SERVER=user@host}
for f in s1/gguf/val_ru.txt s1/gguf/val_chat_ru.txt s1/gguf/e1-TQ2_0.gguf s1/gguf/e2p-TQ2_0.gguf q4b/bench/tern-TQ2_0-head_q6_K.gguf; do
  scp -q "$H:$f" . && echo "ok $f"
done
echo DONE
