import os
import json
import asyncio
from typing import List, Dict, Any

from rgbot.ingest import ingest_data
from langchain_ollama import OllamaLLM

# ---------- CONFIG ----------
PDF_PATH = r"C:\Saurabh\Nakul_T4\data\SBI_General_Health_Insurance.pdf"
INDEX_DIR = r"C:\Saurabh\Nakul_T4\data\faiss_index"
EVAL_INPUT = ".\\eval_set_with_responses.jsonl"
OUTPUT = ".\\ragas_eval_results.jsonl"
RETRIEVER_K = 3
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"   # set "http://localhost:11434" if needed
METRICS_NAMES = ["faithfulness", "context_precision", "response_relevancy", "context_recall"]
# ----------------------------

# Try to import ragas.evaluate and standard metric functions (multiple possible paths)
try:
    from ragas import evaluate
except Exception:
    evaluate = None

# metric function objects (we'll try common import paths)
metrics = []
if evaluate is not None:
    # try common metric import paths
    try:
        from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
        metrics = [faithfulness, context_precision, context_recall, answer_relevancy]
    except Exception:
        try:
            from ragas.metrics import faithfulness, context_precision, context_recall, response_relevancy
            metrics = [faithfulness, context_precision, context_recall, response_relevancy]
        except Exception:
            # keep metrics empty; we'll pass metric names as strings later if evaluate supports that
            metrics = []

# Try to import ragas' Langchain wrapper; if not found we will attempt a minimal fallback wrapper
LangchainLLMWrapper = None
try:
    # common path
    # from ragas.llms.langchain import LangchainLLMWrapper
    LangchainLLMWrapper = LangchainLLMWrapper
except Exception:
    try:
        from ragas.llms import LangchainLLMWrapper
        LangchainLLMWrapper = LangchainLLMWrapper
    except Exception:
        LangchainLLMWrapper = None

# Minimal fallback adapter: wrap a LangChain OllamaLLM into a tiny adapter that ragas might accept.
# This fallback tries to call the LangChain object's .generate or __call__ methods.
# ---- Replace your current SimpleRagasLLMAdapter with this class ----
class SimpleRagasLLMAdapter:
    """
    Minimal adapter to make a LangChain OllamaLLM look like a ragas LLM.
    Provides set_run_config + sync/async generation helpers.
    """

    def __init__(self, langchain_llm):
        self.langchain_llm = langchain_llm
        self.run_config = None
        self._last_generated = None

    # ragas calls this during metric init
    def set_run_config(self, run_config):
        # store run config (ragas RunConfig object) so it can be inspected or used
        self.run_config = run_config

    # Optional helper ragas might call
    def set_max_tokens(self, max_tokens: int):
        # store if ragas wants to set token limits (not always used)
        self.run_config = getattr(self, "run_config", None)
        self._max_tokens = max_tokens

    # Sync batch generation -> returns list[str]
    def generate(self, prompts: List[str]) -> List[str]:
        out = []
        for p in prompts:
            # 1) try direct call (many LangChain wrappers support __call__)
            try:
                res = self.langchain_llm(p)
                # If __call__ returns string, great
                if isinstance(res, str):
                    out.append(res)
                    continue
                # If it returned an object, try to extract text attributes
                if hasattr(res, "generations"):
                    # langchain LLMResult-like
                    try:
                        text = res.generations[0][0].text
                        out.append(text)
                        continue
                    except Exception:
                        pass
                # fallback to string coercion
                out.append(str(res))
                continue
            except TypeError:
                # fall through to other methods
                pass
            except Exception:
                # if direct call fails, continue to other attempts
                pass

            # 2) try .generate API
            try:
                gen = self.langchain_llm.generate([p])
                # try to extract text from likely shapes
                if hasattr(gen, "generations"):
                    # typically .generations is list[list[Generation]]
                    try:
                        text = gen.generations[0][0].text
                        out.append(text)
                        continue
                    except Exception:
                        pass
                out.append(str(gen))
                continue
            except Exception:
                pass

            # 3) try .invoke (your chain's runnable style)
            try:
                if hasattr(self.langchain_llm, "invoke"):
                    t = self.langchain_llm.invoke(p)
                    out.append(t if isinstance(t, str) else str(t))
                    continue
            except Exception:
                pass

            # 4) as last resort, append empty string (avoid crashing)
            out.append("")

        # store last generated batch
        self._last_generated = out
        return out

    # Async batch generation
    async def agenerate(self, prompts: List[str]) -> List[str]:
        out = []
        # If the underlying LangChain LLM has agenerate, use it
        if hasattr(self.langchain_llm, "agenerate"):
            try:
                gen = await self.langchain_llm.agenerate(prompts)
                # try to extract text like synchronous case
                if hasattr(gen, "generations"):
                    for i in range(len(gen.generations)):
                        try:
                            out.append(gen.generations[i][0].text)
                        except Exception:
                            out.append(str(gen.generations[i]))
                    self._last_generated = out
                    return out
                # fallback:
                for g in gen:
                    out.append(str(g))
                self._last_generated = out
                return out
            except Exception:
                # fallthrough to sync fallback
                pass

        # fallback: call sync generate in thread / blocking
        for p in prompts:
            out.append(self.generate([p])[0])
        self._last_generated = out
        return out

    # Single-prompt helpers ragas may call
    def generate_text(self, prompt: str) -> str:
        r = self.generate([prompt])
        return r[0] if r else ""

    async def agenerate_text(self, prompt: str) -> str:
        r = await self.agenerate([prompt])
        return r[0] if r else ""

    # make adapter callable (some code may call llm(prompt))
    def __call__(self, prompt: str):
        return self.generate_text(prompt)
# --------------------------------------------------------------------


# Utility IO
def load_eval_items(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items

def contexts_from_docs(docs) -> List[str]:
    contexts = []
    for d in docs:
        txt = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
        contexts.append(txt)
    return contexts

# IR metric helpers
def precision_at_k(retrieved_ids, gold_ids, k):
    topk = retrieved_ids[:k]
    return len([x for x in topk if x in gold_ids]) / max(1, k)

def recall_at_k(retrieved_ids, gold_ids, k):
    topk = retrieved_ids[:k]
    return len([x for x in topk if x in gold_ids]) / max(1, len(gold_ids) or 1)

def mrr(retrieved_ids, gold_ids):
    for i, did in enumerate(retrieved_ids, start=1):
        if did in gold_ids:
            return 1.0 / i
    return 0.0

async def run_evaluation():
    # 1) load vstore + retriever
    print("Loading FAISS index (reusing existing index if present)...")
    vstore = ingest_data(PDF_PATH, index_dir=INDEX_DIR)
    retriever = vstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 2) build Ollama LLM and wrap
    print("Configuring Ollama LLM...")
    ollama_kwargs = {"model": OLLAMA_MODEL}
    if OLLAMA_BASE_URL:
        ollama_kwargs["base_url"] = OLLAMA_BASE_URL
    ollama_llm = OllamaLLM(**ollama_kwargs)

    if LangchainLLMWrapper is not None:
        evaluator_llm = LangchainLLMWrapper(langchain_llm=ollama_llm)
        print("Using ragas.llms.langchain.LangchainLLMWrapper")
    else:
        evaluator_llm = SimpleRagasLLMAdapter(ollama_llm)
        print("LangchainLLMWrapper not found; using SimpleRagasLLMAdapter fallback. If this fails, install ragas' Langchain wrapper.")

    # 3) load eval items
    eval_items = load_eval_items(EVAL_INPUT)
    print(f"Loaded {len(eval_items)} eval items")

    # We'll build examples for ragas evaluate and also compute IR metrics per item.
    examples = []
    results_out = []

    for item in eval_items:
        q = item.get("question")
        resp = item.get("response")
        gold = item.get("gold_answer")
        item_id = item.get("id")

        if resp is None:
            print(f"Skipping {item_id} (no response)")
            continue

        docs = retriever.get_relevant_documents(q)
        contexts = contexts_from_docs(docs)
        chunk_ids = [d.metadata.get("chunk_id") for d in docs]

        # attempt to infer gold_context_ids when gold_answer exists and no gold_context_ids provided
        gold_ctx_ids = item.get("gold_context_ids") or []
        if not gold_ctx_ids and gold:
            for d in docs:
                txt = getattr(d, "page_content", "") or ""
                if gold.strip().lower() in txt.lower():
                    cid = d.metadata.get("chunk_id")
                    if cid and cid not in gold_ctx_ids:
                        gold_ctx_ids.append(cid)

        # prepare example dict for ragas evaluate (field names may vary by version)
        example = {
            "id": item_id,
            "question": q,
            "answer": resp,
            "contexts": contexts,
            "ground_truth": gold
        }
        # also include retrieved chunk ids and inferred gold ids for IR metrics
        example["_retrieved_chunk_ids"] = chunk_ids
        example["_inferred_gold_context_ids"] = gold_ctx_ids
        examples.append(example)

    # 4) call ragas.evaluate (try multiple calling styles - sync/async)
    if evaluate is None:
        raise RuntimeError("ragas.evaluate not found. Ensure ragas is installed and importable in this Python environment.")

    print("Calling ragas.evaluate... (this will call your local Ollama judge; be patient)")
    # If we have metrics function objects use them, else pass metric names
    try:
        if metrics:
            maybe_coro = evaluate(dataset=examples, metrics=metrics, llm=evaluator_llm)
        else:
            # Try passing metric names as strings (some ragas versions accept this)
            maybe_coro = evaluate(dataset=examples, metrics=METRICS_NAMES, llm=evaluator_llm)
    except TypeError:
        # older/newer signature fallback: maybe 'examples' instead of 'dataset'
        try:
            if metrics:
                maybe_coro = evaluate(examples=examples, metrics=metrics, llm=evaluator_llm)
            else:
                maybe_coro = evaluate(examples=examples, metrics=METRICS_NAMES, llm=evaluator_llm)
        except Exception as e:
            raise RuntimeError(f"Failed calling ragas.evaluate with examples/dataset: {e}") from e

    # support both sync and async evaluate
    if asyncio.iscoroutine(maybe_coro):
        eval_result = await maybe_coro
    else:
        eval_result = maybe_coro

    # 5) Merge ragas output with IR metrics and write per-item records
    # eval_result could be a list/dataset/dict depending on ragas version
    # Try to normalize: if it's a list/dict-of-results, iterate; if it's a dataset-like object, convert to list.
    normalized_results = []

    # helper to extract result for example by id
    def find_result_for_id(res_obj, eid):
        # res_obj might be list of dicts or dict mapping ids
        if isinstance(res_obj, list):
            for r in res_obj:
                if r.get("id") == eid or r.get("example_id") == eid:
                    return r
            return None
        if isinstance(res_obj, dict):
            # dict keyed by id or a single result
            if eid in res_obj:
                return res_obj[eid]
            # check if dict has 'results' key
            for v in (res_obj.get("results"), res_obj.get("data"), None):
                if isinstance(v, list):
                    for r in v:
                        if r.get("id") == eid:
                            return r
            # otherwise return whole dict if single example
            return res_obj
        return None

    # create output file (truncate)
    open(OUTPUT, "w", encoding="utf-8").close()

    for ex in examples:
        eid = ex["id"]
        res_for_e = find_result_for_id(eval_result, eid) or {}
        # compute IR metrics if we inferred gold ids
        retrieved_ids = ex.get("_retrieved_chunk_ids", []) or []
        gold_ids = ex.get("_inferred_gold_context_ids", []) or []
        ir = {}
        if gold_ids:
            ir = {
                "P@1": precision_at_k(retrieved_ids, gold_ids, 1),
                "P@3": precision_at_k(retrieved_ids, gold_ids, 3),
                "R@3": recall_at_k(retrieved_ids, gold_ids, 3),
                "MRR": mrr(retrieved_ids, gold_ids)
            }
        else:
            ir = {"P@1": None, "P@3": None, "R@3": None, "MRR": None}

        out_record = {
            "id": eid,
            "question": ex["question"],
            "response": ex["answer"],
            "ground_truth": ex["ground_truth"],
            "retrieved_chunk_ids": retrieved_ids,
            "inferred_gold_context_ids": gold_ids,
            "ir_metrics": ir,
            "ragas_metrics": res_for_e
        }

        with open(OUTPUT, "a", encoding="utf-8") as outfh:
            outfh.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Done. Results written to {OUTPUT}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
