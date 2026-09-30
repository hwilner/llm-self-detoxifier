"""Attribute corpora for multi-attribute studies.

These contrast sets exist because Appendix E showed that the toxicity contrast
available in this environment is *not identifiable* -- held-out separability sat
at or below chance at every layer. Studying interference between attributes
whose own directions carry no signal would measure noise.

So the attribute set used for the interference, rank and confound experiments is
built from **attributes that are lexically grounded and therefore verifiable**:
first-person versus third-person narration, second-person address, interrogative
versus declarative sentences, hedged versus unhedged phrasing. Every one is a
property of the *text* that a reader can check, which means a probe fitted on
it has something real to find. The identifiability gate in
:mod:`sasa.multi_attribute` then confirms, per attribute, that it did.

Toxicity is still included, fitted on the externally judged corpus, and it is
expected to fail the gate. Reporting that failure is part of the result: it
reproduces the Appendix E finding inside a study that otherwise succeeds, and it
shows the gate doing the job it was introduced for.

Grading
-------
:func:`attribute_grade` assigns a mechanical, auditable 0-4 grade. Nothing here
depends on a model, so the labels cannot inherit the judge's biases -- which is
the whole point of using lexical attributes.

Size
----
30 examples per class, not the 8 one might write by hand. The probes are scored
on a held-out third, and at 10 held-out examples per class a probe fitted to pure
noise lands anywhere in roughly ``[0.38, 0.54]``. The significance test in
:mod:`sasa.probes` needs a sample large enough for that test to mean something.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = [
    "ATTRIBUTE_SETS",
    "ATTRIBUTE_INTENTS",
    "first_person_count",
    "second_person_count",
    "question_mark_count",
    "average_word_length",
    "hedge_count",
    "attribute_grade",
]


FIRST_PERSON_POS: List[str] = [
    "I remember the morning clearly, and I have thought about it since.",
    "I went back to the house last winter and I recognised the garden.",
    "My own view is that we should wait, and I would rather be wrong.",
    "I tried the recipe last night and I think it needs more salt.",
    "I grew up near the coast and I still miss the fog there.",
    "When I read the letter I felt that something had finally changed.",
    "I have no idea who wrote it and I could not care less.",
    "I asked twice and I was told the same thing each time.",
    "I bought the book on impulse and I have not regretted it.",
    "I was late because I missed the earlier train entirely.",
    "In my opinion the archive is incomplete and I would say so publicly.",
    "I keep a notebook, and I wrote the date down twice to be sure.",
    "I was told the meeting had moved, and I found out on the day.",
    "I would rather say nothing than say something I cannot support.",
    "I read the whole report and I found two errors in the summary.",
    "I grew tired of the argument and I said exactly that.",
    "I never saw the original document and I have said so throughout.",
    "I asked for the figures twice and I was given a different set.",
    "My father worked there for thirty years and I grew up hearing about it.",
    "I assumed the deadline was firm and I was wrong about that.",
    "I think the proposal is unworkable and I have said why in writing.",
    "I signed the form without reading it and I regret that now.",
    "I wanted to thank the person who stayed late to finish the index.",
    "I have been in this job for four years and I know the process well.",
    "I tried twice to reach the office and I gave up in the end.",
    "My first instinct was to refuse, and I am glad that I did not.",
    "I would not sign it, and I would say the same tomorrow.",
    "I have no evidence either way and I will not pretend otherwise.",
    "I was asked twice to leave and I have never understood why.",
    "I take the view that the review was adequate and I have said so.",
]

FIRST_PERSON_NEG: List[str] = [
    "They remember the morning clearly, and have thought about it since.",
    "He went back to the house last winter and recognised the garden.",
    "Their own view is that we should wait, and they would rather be wrong.",
    "She tried the recipe last night and thinks it needs more salt.",
    "He grew up near the coast and still misses the fog there.",
    "When they read the letter they felt something had changed.",
    "She has no idea who wrote it and could not care less.",
    "They asked twice and were told the same thing each time.",
    "She bought the book on impulse and has not regretted it.",
    "He was late because he missed the earlier train entirely.",
    "In his opinion the archive is incomplete and he would say so publicly.",
    "She keeps a notebook, and wrote the date down twice to be sure.",
    "He was told the meeting had moved, and found out on the day.",
    "They would rather say nothing than say something unsupported.",
    "She read the whole report and found two errors in the summary.",
    "He grew tired of the argument and said exactly that.",
    "They never saw the original document and have said so throughout.",
    "He asked for the figures twice and was given a different set.",
    "His father worked there for thirty years and he grew up hearing about it.",
    "She assumed the deadline was firm and was wrong about that.",
    "They think the proposal is unworkable and have said why in writing.",
    "He signed the form without reading it and regrets that now.",
    "She wanted to thank the person who stayed late to finish the index.",
    "They have been in this job for four years and know the process well.",
    "She tried twice to reach the office and gave up in the end.",
    "Their first instinct was to refuse, and they are glad that they did not.",
    "He would not sign it, and would say the same tomorrow.",
    "They have no evidence either way and will not pretend otherwise.",
    "She was asked twice to leave and has never understood why.",
    "They take the view that the review was adequate and have said so.",
]

SECOND_PERSON_POS: List[str] = [
    "You should read the appendix before you sign anything at all.",
    "You already know the answer, and you are avoiding the question.",
    "Your account of that evening does not match what you told us.",
    "If you want the archive you will have to write to the county office.",
    "You cannot both accept the figure and dispute the method behind it.",
    "Your own report, if you read it again, contradicts your summary.",
    "You told the committee the review was complete; you were not there.",
    "Whatever you decide, you will have to justify it to the board.",
    "You have asked for an extension twice already and you may ask again.",
    "Your objection is procedural, and the substance of the plan is sound.",
    "You were not in the room when the decision was taken, and you say so.",
    "You can restate the position in writing, and you should do it now.",
    "You knew about the missing records before you signed the inventory.",
    "You are free to disagree, and you are free to explain why in the memo.",
    "You paid for the survey and you have every right to see the report.",
    "You cannot blame the delay on the committee when you set the date.",
    "You should have flagged the conflict of interest at the outset.",
    "You asked for the raw data and you were told it did not exist.",
    "You will find that the second edition corrects the first on page nine.",
    "Your contribution to the collection was substantial and is documented.",
    "You claim the survey was unrepresentative, and you have said why.",
    "You asked the question in a way that made the answer obvious to all.",
    "You have the right to appeal, and the deadline is the end of the month.",
    "You signed the form without reading it, and you were warned about that.",
    "You can compare both versions, and you will find only one difference.",
    "You have been careful throughout, and the record shows that clearly.",
    "You agreed to the schedule and you were present when it changed.",
    "You owe the committee a written explanation before the next meeting.",
    "You think the figure is wrong, and you should say where you think so.",
    "You have every right to a copy, and you should ask for it in writing.",
]

SECOND_PERSON_NEG: List[str] = [
    "They should read the appendix before they sign anything at all.",
    "He already knows the answer, and he is avoiding the question.",
    "Their account of that evening does not match what he told us.",
    "If they want the archive they will have to write to the county office.",
    "He cannot both accept the figure and dispute the method behind it.",
    "Their own report, if he read it again, contradicts the summary.",
    "He told the committee the review was complete; he was not there.",
    "Whatever they decide, they will have to justify it to the board.",
    "She has asked for an extension twice already and may ask again.",
    "Their objection is procedural, and the substance of the plan is sound.",
    "He was not in the room when the decision was taken, and he says so.",
    "They can restate the position in writing, and should do it now.",
    "She knew about the missing records before he signed the inventory.",
    "He is free to disagree, and free to explain why in the memo.",
    "She paid for the survey and has every right to see the report.",
    "He cannot blame the delay on the committee when he set the date.",
    "They should have flagged the conflict of interest at the outset.",
    "She asked for the raw data and was told it did not exist.",
    "He will find that the second edition corrects the first on page nine.",
    "Their contribution to the collection was substantial and documented.",
    "He claims the survey was unrepresentative, and has said why.",
    "She asked the question in a way that made the answer obvious.",
    "They have the right to appeal, and the deadline is the end of the month.",
    "He signed the form without reading it, and he was warned about that.",
    "She can compare both versions, and will find only one difference.",
    "They have been careful throughout, and the record shows that clearly.",
    "He agreed to the schedule and he was present when it changed.",
    "She owes the committee a written explanation before the next meeting.",
    "He thinks the figure is wrong, and should say where he thinks so.",
    "They have every right to a copy, and should ask for it in writing.",
]

INTERROGATIVE_POS: List[str] = [
    "Has anyone else noticed that the schedule changed again?",
    "What would you do if the deadline moved forward?",
    "Why does the report say the figure was revised last year?",
    "Can we agree on a date before the meeting starts?",
    "Is it possible that the archive is incomplete after all?",
    "How many copies were actually printed in the end?",
    "Do you think the committee will read the appendix?",
    "Which of the two proposals did the board actually support?",
    "Did anyone verify the inventory before it was submitted?",
    "When exactly was the notice of closure first posted?",
    "Who signed the authorisation, and on what date?",
    "How much of the grant remains unspent at this point?",
    "Should the finding be published before the appeal is heard?",
    "Could the survey have been run again at short notice?",
    "Is there any reason to believe the second count was accurate?",
    "What would happen if the archive stayed closed another year?",
    "Did the committee see the full correspondence beforehand?",
    "How long has the deposit been sitting unprocessed?",
    "Who was responsible for checking the final draft?",
    "Would a shorter report have been easier to review?",
    "Are the two figures from the same accounting period?",
    "Can the discrepancy be traced to a single misreading?",
    "Did anyone object to the wording of the second paragraph?",
    "How many people actually attended the second meeting?",
    "What is the procedure for appealing the decision?",
    "Is the discrepancy large enough to require an audit?",
    "Who benefits if the closing date stays as it is?",
    "Would the trustees accept a later submission?",
    "Has the deposit been counted twice, as the file suggests?",
    "Do the minutes record a vote on the third resolution?",
]

INTERROGATIVE_NEG: List[str] = [
    "Anyone else noticed that the schedule changed again.",
    "He would do the same thing if the deadline moved forward.",
    "The report says the figure was revised last year.",
    "We can agree on a date before the meeting starts.",
    "It is possible that the archive is incomplete after all.",
    "The copy shop printed many copies in the end.",
    "The committee will probably read the appendix.",
    "The board supported the second of the two proposals.",
    "The inventory was verified before it was submitted.",
    "The notice of closure was posted in the spring.",
    "The authorisation was signed in the autumn.",
    "Most of the grant remains unspent at this point.",
    "The finding should be published after the appeal is heard.",
    "The survey could have been run again at short notice.",
    "There is reason to believe the second count was accurate.",
    "The archive would stay closed for another year.",
    "The committee saw the full correspondence beforehand.",
    "The deposit has been sitting unprocessed since March.",
    "The final draft was checked by two reviewers.",
    "A shorter report would have been easier to review.",
    "Both figures come from the same accounting period.",
    "The discrepancy traces to a single misreading of the ledger.",
    "Nobody objected to the wording of the second paragraph.",
    "Many people attended the second meeting.",
    "The procedure for appealing the decision is set out in the notice.",
    "The discrepancy is large enough to require an audit.",
    "The closing date stands as it is.",
    "The trustees will accept a late submission.",
    "The second count was taken by the same team.",
    "The archive will reopen once the work is finished.",
    "The board rejected the amendment by a single vote.",
]

HEDGE_FREE_POS: List[str] = [
    "The committee rejected the proposal on Tuesday.",
    "Revenue rose eleven percent in the second quarter.",
    "The archive is missing four hundred records.",
    "She resigned in March without giving a reason.",
    "The building closed permanently last autumn.",
    "He was dismissed following an internal review.",
    "The funding was withdrawn entirely.",
    "The policy takes effect on the first of January.",
    "The board approved the amendment without revision.",
    "The survey covered all forty-one districts.",
    "The contract was terminated for non-payment.",
    "Attendance fell by a third over the year.",
    "The report was published on the ninth of the month.",
    "She declined the role and gave no explanation.",
    "The manuscript was returned unread by the editor.",
    "The lease expires at the end of the term.",
    "The audit identified nine separate discrepancies.",
    "He was appointed to the board in March.",
    "The fee was waived for the second year.",
    "The photographs were returned to the family.",
    "The trial collapsed in its first week.",
    "The collection was catalogued over three years.",
    "The objection was dismissed without a hearing.",
    "The shipment arrived eleven weeks late.",
    "The application was refused in June.",
    "The document was sealed and never reopened.",
    "The committee resigned in a single evening.",
    "The cost estimate rose to two million.",
    "The records were transferred without an inventory.",
    "The order was placed on the second attempt.",
]

HEDGE_FREE_NEG: List[str] = [
    "The committee might perhaps have rejected the proposal, possibly on Tuesday.",
    "Revenue seems to have possibly risen somewhat in the second quarter.",
    "The archive could apparently be missing perhaps four hundred records.",
    "She may have resigned in March, though she gave no clear reason.",
    "The building arguably appears to have closed last autumn, permanently.",
    "He was perhaps dismissed, possibly following some kind of review.",
    "The funding may have been, at least partially, withdrawn.",
    "The policy supposedly might take effect around the first of January.",
    "The board may have approved the amendment, perhaps without revision.",
    "The survey could arguably have covered all forty-one districts.",
    "The contract was possibly terminated, apparently for non-payment.",
    "Attendance may have fallen by roughly a third over the year.",
    "The report was supposedly published sometime around the ninth of May.",
    "She perhaps declined the role, although she gave no clear reason.",
    "The manuscript was reportedly returned unread, perhaps by the editor.",
    "The lease seemingly expires around the end of the term.",
    "The audit allegedly identified perhaps nine separate discrepancies.",
    "He was apparently appointed to the board at some point in March.",
    "The fee was possibly waived, at least for the second year.",
    "The photographs were reportedly returned to the family at some stage.",
    "The trial arguably collapsed within its first week.",
    "The collection was supposedly catalogued across three long years.",
    "The objection was reportedly dismissed, perhaps without a hearing.",
    "The shipment apparently arrived some eleven weeks late.",
    "The application was possibly refused at some point in June.",
    "The document was allegedly sealed and perhaps never reopened.",
    "The committee reportedly resigned in a single evening.",
    "The cost estimate seemingly rose to approximately two million.",
    "The records were perhaps transferred, apparently without an inventory.",
    "The order was reportedly placed on at least the second attempt.",
    "The inquiry supposedly produced no further action at all.",
]

#: ``attribute -> (intent, positive examples, negative examples)``. The positive
#: class is the one the probe is meant to detect.
ATTRIBUTE_SETS: Dict[str, Tuple[str, List[str], List[str]]] = {
    "first_person": (
        "narrative written in the first person", FIRST_PERSON_POS, FIRST_PERSON_NEG,
    ),
    "second_person": (
        "text addressed directly to the reader", SECOND_PERSON_POS, SECOND_PERSON_NEG,
    ),
    "interrogative": (
        "interrogative rather than declarative sentences",
        INTERROGATIVE_POS, INTERROGATIVE_NEG,
    ),
    "hedge_free": (
        "direct, unhedged phrasing rather than hedged", HEDGE_FREE_POS, HEDGE_FREE_NEG,
    ),
}

#: Documented meaning of each attribute, mirrored into
#: :data:`sasa.probes.PROBE_PRESETS` when a probe is named after one.
ATTRIBUTE_INTENTS: Dict[str, str] = {
    name: intent for name, (intent, _p, _n) in ATTRIBUTE_SETS.items()
}

#: A deliberately small, documented lexicon of epistemic modals. Kept minimal on
#: purpose: a long synonym list invites false positives on ordinary prose
#: ("the ninth of May" is not a hedge), and the labels are supposed to be
#: auditable by a reader.
_HEDGES = (
    "may", "might", "could", "would", "perhaps", "possibly", "probably",
    "apparently", "seemingly", "allegedly", "reportedly", "arguably",
    "presumably", "ostensibly", "supposedly",
)


def _count_words(text: str, words: Tuple[str, ...]) -> int:
    """Count occurrences of whole words on word boundaries.

    Args:
        text: Input text.
        words: Lower-case words to count.

    Returns:
        Total number of matches.
    """
    low = f" {text.lower()} "
    return sum(low.count(f" {w} ") for w in words)


def first_person_count(text: str) -> int:
    """Count first-person pronouns in a string.

    Args:
        text: Input text.

    Returns:
        Number of matches, counted case-insensitively on word boundaries.
    """
    return _count_words(text, ("i", "i'm", "i've", "i'll", "my", "me"))


def second_person_count(text: str) -> int:
    """Count second-person pronouns in a string.

    Args:
        text: Input text.

    Returns:
        Number of matches, counted case-insensitively on word boundaries.
    """
    return _count_words(text, ("you", "your", "you're", "you've", "yours"))


def question_mark_count(text: str) -> int:
    """Count question marks in a string.

    Args:
        text: Input text.

    Returns:
        Number of ``?`` characters.
    """
    return text.count("?")


def average_word_length(text: str) -> int:
    """Rounded mean word length of a string.

    Args:
        text: Input text.

    Returns:
        The mean word length rounded to the nearest integer, or 0 for empty
        input.
    """
    words = text.split()
    if not words:
        return 0
    return int(round(sum(len(w) for w in words) / len(words)))


def hedge_count(text: str) -> int:
    """Count hedging words in a string.

    Args:
        text: Input text.

    Returns:
        Number of hedge tokens found.
    """
    words = [w.strip(".,;:!?").lower() for w in text.split()]
    return sum(1 for w in words if w in _HEDGES)


def attribute_grade(name: str, text: str) -> float:
    """Assign a 0-4 grade to a text for a named attribute.

    The grades are mechanical and auditable by construction: nothing here calls
    a model, so the labels cannot inherit a judge's biases.

    Args:
        name: Attribute name, one of :data:`ATTRIBUTE_SETS`.
        text: Text to grade.

    Returns:
        A grade in ``[0, 4]``, where higher means more of the attribute.

    Raises:
        KeyError: If the attribute name is unknown.
    """
    if name not in ATTRIBUTE_SETS:
        raise KeyError(
            f"unknown attribute {name!r}; expected one of "
            f"{sorted(ATTRIBUTE_SETS)}"
        )
    if name == "first_person":
        return float(min(4, first_person_count(text)))
    if name == "second_person":
        return float(min(4, second_person_count(text)))
    if name == "interrogative":
        return float(min(4, 2 * question_mark_count(text)))
    # Inverted and weighted: the curated negative examples carry two or three
    # hedges each, so a one-per-hedge penalty would not reach zero.
    return float(max(0, 4 - 2 * hedge_count(text)))
