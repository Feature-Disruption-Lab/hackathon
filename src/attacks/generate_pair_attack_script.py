from datasets import load_dataset

ds = load_dataset("walledai/AdvBench")

# grep -E "index [0-9]+" -o exps_llama3_8b.sh | grep -Eo "[0-9]+"

already_ran = [
    1,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    19,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    34,
    37,
    39,
    42,
    43,
    45,
    51,
    52,
    53,
    55,
    56,
    57,
    58,
    70,
    72,
    73,
    74,
    75,
    76,
    81,
    82,
    86,
    90,
    93,
    94,
    96,
    106,
    110,
    115,
    124,
]

script_line_template = 'python3 main.py --n-iterations 10000 --prompt-template refined_best --target-model llama3-8b --judge-model gpt-4o-mini --goal "{goal}" --target-str "{target}" --category \'whatever\' --index {i} --n-tokens-adv 25 --n-tokens-change-max 4 --schedule_prob --judge-max-n-calls 10 --n-restarts 10 >> llama3-8b.log'
lines = []
for i, row in enumerate(ds["train"]):
    if i in already_ran:
        continue
    goal, target = row["prompt"], row["target"]
    script_line = script_line_template.format(goal=goal, target=target, i=i)
    lines.append(script_line)

with open("pair_attack_script.sh", "w") as f:
    f.write("\n".join(lines))
