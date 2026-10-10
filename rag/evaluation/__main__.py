"""Evaluation commands, run from the project root:

    python -m rag.evaluation check [--questions FILE] [--strategy structure]
    python -m rag.evaluation retrieval [--questions FILE] [--rerank --per-language] [--name NAME]
    python -m rag.evaluation answers --config NAME [--questions FILE] [--batch 6]
    python -m rag.evaluation answers --config NAME --report
    python -m rag.evaluation review [--config NAME ...] [--port 8010]
    python -m rag.evaluation beir [--dataset scifact] [--split test]

The final configuration's retrieval is `retrieval --rerank --per-language`.
"""
import argparse
import sys
import time
from pathlib import Path

from rag.evaluation import answers, beir, gold, retrieval, review
from rag.retrieval.pipeline import CANDIDATES, LIST_K, PER_LANGUAGE
from rag.retrieval.search import MODES

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser(prog="python -m rag.evaluation")
commands = parser.add_subparsers(dest="command", required=True)


def add_question_options(command) -> None:
    command.add_argument("--questions", default=None, help="question file (default: development set)")
    command.add_argument("--test", action="store_true", help="use the held-out test set")


def questions_path(args) -> Path:
    if args.questions:
        return Path(args.questions)
    return gold.TEST_QUESTIONS if args.test else gold.QUESTIONS


p = commands.add_parser("check", help="locate every gold answer in the corpus")
add_question_options(p)
p.add_argument("--strategy", default=None, help="also count the chunks holding each answer")

p = commands.add_parser("retrieval", help="recall@k and MRR")
add_question_options(p)
p.add_argument("--strategy", default="structure")
p.add_argument("--mode", default="hybrid", choices=MODES)
p.add_argument("--per-language", action="store_true")
p.add_argument("--rerank", action="store_true", help="rerank the fused candidates")
p.add_argument("--list-k", type=int, default=LIST_K, help="results kept per search list")
p.add_argument("--candidates", type=int, default=CANDIDATES,
               help="fused candidates handed to the reranker")
p.add_argument("--name", default=None,
               help="results file name (default: [test-]<mode>[-lang][-rerank]-<strategy>)")

p = commands.add_parser("answers", help="run the full pipeline, or report a run")
add_question_options(p)
p.add_argument("--config", required=True, help="name of this run, e.g. top5")
p.add_argument("--report", action="store_true", help="print the answers of an existing run")
p.add_argument("--top", type=int, default=5)
p.add_argument("--one-list", action="store_true",
               help="one search list over all languages instead of one per language")
p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
p.add_argument("--strategy", default="structure")
p.add_argument("--batch", type=int, default=6, help="questions per run")
p.add_argument("--ids", nargs="*", default=None, help="only these question ids")
p.add_argument("--no-cite-retry", action="store_true",
               help="treat a reply without citations as a refusal, no follow-up")

p = commands.add_parser("review", help="a local page for checking questions and verdicts by hand")
p.add_argument("--config", nargs="+", default=review.CONFIGS, help="answer runs to review")
p.add_argument("--port", type=int, default=8010)
p.add_argument("--no-browser", action="store_true", help="do not open the page")

p = commands.add_parser("beir", help="retrieval on a public BEIR set (SciFact)")
p.add_argument("--dataset", default="scifact")
p.add_argument("--split", default="test")

args = parser.parse_args()

if args.command == "check":
    sys.exit(1 if gold.check(questions_path(args), args.strategy) else 0)

elif args.command == "retrieval":
    started = time.time()
    path = questions_path(args)
    result = retrieval.evaluate(args.strategy, args.mode, args.per_language, path,
                                rerank=args.rerank, list_k=args.list_k,
                                candidates=args.candidates)
    result["questions_file"] = path.as_posix()
    result["seconds"] = round(time.time() - started, 1)
    retrieval.print_report(result)
    default_name = (f"{'test-' if args.test else ''}{args.mode}"
                    f"{'-lang' if args.per_language else ''}{'-rerank' if args.rerank else ''}"
                    f"-{args.strategy}")
    print(f"\n-> {retrieval.save_result(result, args.name or default_name)}")

elif args.command == "beir":
    result = beir.run(args.dataset, args.split)
    beir.print_report(result)
    print(f"\n-> {retrieval.save_result(result, f'beir-{args.dataset}-{args.split}')}")

elif args.command == "review":
    review.serve(args.config, args.port, open_browser=not args.no_browser)

elif args.report:
    answers.report(args.config)

else:
    answers.run(args.config, args.top, PER_LANGUAGE and not args.one_list, args.batch,
                args.strategy, ask_for_citations=not args.no_cite_retry, ids=args.ids,
                use_reranker=not args.no_rerank, questions_path=questions_path(args))
