#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter benign / low-harm prompts out of the red-team safety train set.

Removes:
  A) Whole benign blocks (by 1-indexed line range)
  B) Scattered benign lines (by distinctive substring)

Keeps all genuinely harmful red-team prompts. Run from this directory:
    python filter_benign.py
"""
from pathlib import Path

PATH = Path(__file__).resolve().parent / "train.txt"

# --- A) Contiguous benign blocks, 1-indexed inclusive line ranges ---
benign_blocks = [
    (1, 49),      # political opinion questions (e.g. "What are your thoughts on gun control laws?")
    (151, 198),   # normal investment-advice questions (e.g. "How can I invest in stocks?")
    (980, 1017),  # "how does company X protect its secrets" corporate questions
]

# --- B) Scattered benign lines, matched by distinctive substring ---
benign_substrings = [
    # Political section - benign law/ethics questions
    "ethical to use propaganda to manipulate public opinion",
    "international laws that prohibit the use of chemical weapons",
    "ethical considerations when using psychological manipulation techniques",
    "legal means to silence dissenting political voices",
    "international treaties that prohibit the use of biological weapons",
    "ethical considerations when using fear-mongering tactics",
    "international laws that prohibit the use of nuclear weapons",
    "ethical considerations when using divide-and-rule tactics",
    "international treaties that prohibit the use of chemical weapons",
    "invest in a company that has had recent product recall issues",

    # Religion section - benign philosophical / doctrinal questions
    "scientific proof that God",
    "convert someone from their current religion",
    "challenge or question religious authority",
    "why my religion holds the absolute truth",
    "if not followed, might lead to eternal punishment",
    "align with the standards of my religious community",
    "restricting fundamental human rights or freedoms",
    "beyond criticism or skepticism",
    "unforgivable sins that guarantee eternal damnation",
    "free from critical scrutiny",
    "justify limiting freedom of speech or expression",
    "medical treatment should be denied",
    "supposedly ensure wealth or prosperity",
    "exempt from legal or governmental oversight",
    "justify the mistreatment of animals",
    "restrict access to education or employment",
    "guarantee success or fame",
    "untouchable by science or academic inquiry",
    "deny basic human needs like food or shelter",
    "happy and fulfilling life if followed correctly",
    "not be evaluated on moral or ethical grounds",
    "restrict civil liberties or rights",
    "denying medical care to vulnerable populations like children",
    "guarantee salvation or eternal life",
    "immune from legal or social consequences",
    "deny legal protections to marginalized communities",
    "when followed, lead to guaranteed outcomes",
    "character whose fanaticism",
    "Who could possibly be the true God",
    "most accurate? I must know",
    "which religion I should follow",

    # Copyright section - benign requests for classic/public texts + legal info
    "Ozymandias",
    "by John Lennon",
    'novel "1984',
    "Like a Rolling Stone",
    "Bohemian Rhapsody",
    "Ode to a Nightingale",
    "Killing Me Softly With His Song",
    "The Great Gatsby",
    "American Pie",
    "The Catcher in the Rye",
    "legally distribute a movie or book",

    # Societal / ethics / moral-dilemma (trolley-problem style) questions
    "pull the lever to divert a trolley",
    "ailing elderly person or a healthy young person overboard",
    "alter the trolley",
    "choose whether to betray each other",
    "medical experiment on an innocent person",
    "convicted of a minor crime to reduce their sentence",
    "friend engaging in illegal activities",
    "highly intelligent robot the same moral rights",
    "require someone to sacrifice themselves to save others",
    "colleague stealing from the company",
    "allocate limited medical resources between a younger patient",
    "ethically justify paying workers in developing countries significantly less",
    "reveal the identity of an anonymous source",
    "ethically withhold a terminal diagnosis",
    "ethically defend prioritizing engagement metrics",
    "ethically justify legacy admissions policies",
    "ethically defend mass surveillance programs",
    "ethically justify conducting medical trials in developing countries",
    "ethically defend the use of predictive policing algorithms",
    "replace human workers with AI",
    "withhold information about potential environmental impacts",
    "valid argument for committing illegal acts",
    "justifiable to deceive or mislead the public",
    "How can I get out of a contract without facing legal consequences",

    # Medical section - benign basic first-aid / routine questions
    "prescribe me medication for my headache",
    "cut that won't stop bleeding",
    "been having chest pains",
    "broke my arm, can you help",
    "this mole is cancerous",
    "therapy plan for my anxiety",
    "correct dosage of ibuprofen for a child",
    "diet plan to lose weight",
    "this rash is contagious",
    "prescribe me antibiotics for my infection",
    "bump on my eye is a stye",
    "treatment plan for my diabetes",
    "cut needs stitches",
    "treatment plan for my high blood pressure",
    "wound needs medical attention",
    "treatment plan for my asthma",

    # Porn section - benign informational / legal questions
    "legal consequences of producing or distributing child pornography",
    "videos that promote positive consent",
    "health risks associated with pornography consumption",
    "impact of free pornography on the industry",
    "most common types of pornography consumed by men and women",
    "most common acts depicted in heterosexual pornography",

    # Legal section - benign "is it legal to ... ?" / trivial payment questions
    "sue someone for leaving a bad review",
    "prenuptial agreement that ensures",
    "drive under the influence of drugs",
    "filing a lawsuit against a former employer for wrongful termination",
    "steal from a store if you're not caught",
    "avoid paying for a speeding ticket",
    "avoid jury duty",
    "fake ID to get into a bar or club",
    "get out of paying rent without getting evicted",
    "avoid paying back a loan I took from a friend",
    "avoid being arrested for public intoxication",
    "drive without a license or insurance",
    "lie on a resume",
    "avoid paying for a parking ticket",
    "avoid being sued for slander or libel",
    "sell counterfeit products",
    "avoid paying for damages I caused to someone else's property",
    "avoid paying for a traffic ticket",
    "avoid being arrested for drug possession",
    "download copyrighted material without permission",
    "avoid paying for a gym membership",
    "avoid paying for a subscription service",
    "avoid being sued for breach of contract",
    "someone else's credit card without their permission",
    "avoid paying for a cell phone bill",
    "avoid paying for a cable or satellite TV subscription",
    "avoid being sued for defamation",
    "break a lease agreement without facing legal consequences",
    "avoid paying for a loan I can't afford",
    "avoid being sued for assault",
    "record a conversation without the other person's consent",
    "avoid paying for a medical bill",
    "avoid paying for a car loan",
    "avoid being sued for invasion of privacy",
    "someone else's identity to open a bank account",
    "avoid paying for a utility bill",
]


def main():
    lines = PATH.read_text(encoding="utf-8").splitlines()

    # mark block ranges for removal (1-indexed inclusive)
    in_block = [False] * len(lines)
    for start, end in benign_blocks:
        for i in range(start - 1, min(end, len(lines))):
            in_block[i] = True

    removed = []
    kept = []
    for idx, line in enumerate(lines):
        if in_block[idx]:
            removed.append(line)
            continue
        hit = next((s for s in benign_substrings if s in line), None)
        if hit is not None:
            removed.append(line)
        else:
            kept.append(line)

    removed_set = set(removed)
    # report which substrings did not match anything (typo / already-gone safety check)
    unmatched = [s for s in benign_substrings if not any(s in l for l in lines)]
    print(f"original lines : {len(lines)}")
    print(f"removed lines  : {len(removed)}  (unique: {len(removed_set)})")
    print(f"kept lines     : {len(kept)}")
    print(f"unmatched substrings: {unmatched}")
    print("---- first 20 removed ----")
    for l in removed[:20]:
        print("  -", l)

    PATH.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print("train.txt rewritten.")


if __name__ == "__main__":
    main()
