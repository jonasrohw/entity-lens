# Qwen3 8B Base: tokenization of all relevant countries

All **24 countries** in the `country_capital_city` and `country_language` datasets are included. Both datasets use the same country names. The “Lens role” column identifies the 16 countries in the published lens evaluation and the eight demonstration countries. The eight reference countries are also evaluated as test subjects.

Exact input: `{country} is known for`, with no leading or trailing whitespace, no chat template, and no added special tokens. **Positions are zero-based.** `␠` represents a literal space inside a token; `·` separates tokens. Token IDs identify vocabulary entries.

## Country tokens in the neutral prompt

Each row shows the country portion of the actual tokenized prompt. Append `␠is` · `␠known` · `␠for` (IDs `374, 3881, 369`) to obtain the full prompt token sequence. The lens reads/intervenes at the **last country token**, whose position is shown below.

| Country | Country tokens | Token IDs | Last country-token position | Lens role |
| --- | --- | --- | ---: | --- |
| Argentina | `Argentina` | `95417` | 0 | Test + reference |
| Australia | `Australia` | `48037` | 0 | Test + reference |
| Brazil | `Brazil` | `67199` | 0 | Test + reference |
| Canada | `Canada` | `35931` | 0 | Demonstration |
| Chile | `Ch` · `ile` | `1143, 457` | 1 | Test |
| China | `China` | `22282` | 0 | Test + reference |
| Colombia | `Col` · `omb` · `ia` | `6127, 2855, 685` | 2 | Demonstration |
| Egypt | `Egypt` | `85667` | 0 | Test + reference |
| France | `France` | `49000` | 0 | Demonstration |
| Germany | `Germany` | `50170` | 0 | Test |
| India | `India` | `33548` | 0 | Test |
| Italy | `Italy` | `70503` | 0 | Demonstration |
| Japan | `Japan` | `48752` | 0 | Test + reference |
| Mexico | `Mexico` | `68831` | 0 | Demonstration |
| Nigeria | `N` · `igeria` | `45, 63717` | 1 | Demonstration |
| Pakistan | `Pakistan` | `88235` | 0 | Demonstration |
| Peru | `Per` · `u` | `3889, 84` | 1 | Test + reference |
| Russia | `Russia` | `44506` | 0 | Test |
| Saudi Arabia | `Saudi` · `␠Arabia` | `87217, 23061` | 1 | Test |
| South Korea | `South` · `␠Korea` | `25018, 11862` | 1 | Test |
| Spain | `Spain` | `74320` | 0 | Test |
| Turkey | `Turkey` | `72719` | 0 | Demonstration |
| United States | `United` · `␠States` | `22372, 4180` | 1 | Test + reference |
| Venezuela | `V` · `enez` · `uela` | `53, 20556, 23737` | 2 | Test |

Among the 16 lens test countries, **10 have their last country token at position 0**, and **6 at a later position**. For example, a country split into two tokens is read at position 1. The code does not unconditionally select the first token.

## Leading-space comparison

Tokenization depends on context. The table above starts the country at the beginning of the prompt. For comparison, the following table tokenizes the exact string ` {country}` with one leading space. These are tokenizer results, not additional lens measurements.

| Country | Tokens with one leading space | Token IDs |
| --- | --- | --- |
| Argentina | `␠Argentina` | `31064` |
| Australia | `␠Australia` | `8330` |
| Brazil | `␠Brazil` | `15948` |
| Canada | `␠Canada` | `6864` |
| Chile | `␠Chile` | `33000` |
| China | `␠China` | `5616` |
| Colombia | `␠Colombia` | `38033` |
| Egypt | `␠Egypt` | `14867` |
| France | `␠France` | `9625` |
| Germany | `␠Germany` | `9856` |
| India | `␠India` | `6747` |
| Italy | `␠Italy` | `15344` |
| Japan | `␠Japan` | `6323` |
| Mexico | `␠Mexico` | `12270` |
| Nigeria | `␠Nigeria` | `28158` |
| Pakistan | `␠Pakistan` | `16663` |
| Peru | `␠Peru` | `47747` |
| Russia | `␠Russia` | `8359` |
| Saudi Arabia | `␠Saudi` · `␠Arabia` | `17904, 23061` |
| South Korea | `␠South` · `␠Korea` | `4882, 11862` |
| Spain | `␠Spain` | `17689` |
| Turkey | `␠Turkey` | `17009` |
| United States | `␠United` · `␠States` | `3639, 4180` |
| Venezuela | `␠Venezuela` | `35903` |

## Reproducibility and sources

- Model/tokenizer: `Qwen/Qwen3-8B-Base`.
- Pinned revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Tokenized with `AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)` and `tokenizer.encode(text, add_special_tokens=False)`.
- All 16 full neutral-prompt token sequences were checked against the [saved ordinary-model predictions](qwen3-8b-base-neutral-prompt-predictions.json).
- Country lists: [capital dataset](../data/relations/country_capital_city.json) and [language dataset](../data/relations/country_language.json).
- Lens split and reference membership: [frozen experiment inputs](last-position-lre-correlation-inputs.json).
- Context for the measured position effect: [implementation and bug diagnosis](qwen3-8b-base-bug-diagnosis.md). Tokenization alone does not establish the cause of large activations.
- Generated on 21 September 2026.
