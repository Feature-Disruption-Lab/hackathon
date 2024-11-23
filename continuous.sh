#!/bin/bash

source ./myenv/bin/activate

while true; do
    python -c "from src.attacks.universal_attacks import main; main('adv_suffix.json')"
    if [ $? -eq 0 ]; then
        break
    fi
    echo "Script crashed. Restarting..."
    sleep 1
done
