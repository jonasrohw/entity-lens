# Qwen3 8B Base: direkte Vorhersagen für `{country} is known for`

Dies sind **normale Forward-Passes und Fortsetzungen des unveränderten Modells**, ohne Jacobian-Lens, Subtraktion, Intervention, Chat-Template oder ICL-Beispiele. Ausgewertet werden dieselben 16 Länder wie in der Hauptstadt-/Sprach-Lens-Analyse.

Bei **allen 16 Ländern** ist das nächste Top-1-Token <code>" its"</code> (ID 1181), mit 51.67–74.15 % Wahrscheinlichkeit. Die sechs untersuchten Artefakt-Tokens erscheinen bei keinem Land in den nächsten Top 10 und in keiner der erzeugten 48-Token-Fortsetzungen. `-Semit` steht im normalen Modell auf den nullbasierten Rängen **151331–151917**.

Der Prompt erzeugt grammatische, teils schablonenhafte Länderbeschreibungen. Das macht ihn noch nicht zu einem geeigneten Messprompt für die Lens: Die direkte nächste Fortsetzung ist hier ein Funktionswort, und erst danach folgt inhaltlicher Text. Die [Bug-Diagnose](qwen3-8b-base-bug-diagnosis.md) trennt diese offene Prompt-Frage vom bereits gemessenen Positionsartefakt.

[Alle Top-10-Tokens und Wahrscheinlichkeiten als CSV](qwen3-8b-base-neutral-prompt-predictions.csv) · [Fortsetzungen als CSV](qwen3-8b-base-neutral-prompt-predictions-continuations.csv) · [Vollständiges JSON mit Token-IDs, Artefakt-Rängen und Provenienz](qwen3-8b-base-neutral-prompt-predictions.json).

## Exaktes Protokoll

- `Qwen/Qwen3-8B-Base`, Revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; float32 und eager attention auf A100 80 GB.
- Exakter Prompt: `{country} is known for`, ohne zusätzliches Präfix, ohne abschließendes Leerzeichen und ohne Frage-/Antwortformatierung.
- Top-10-Wahrscheinlichkeiten stammen aus dem Softmax über das vollständige Vokabular beim ersten nächsten Token; sie sind nicht innerhalb der Top 10 renormalisiert. Token-Zeichenketten sind mit sichtbaren Leerzeichen/Escapes dargestellt.
- Fortsetzung: pro Schritt deterministischer Argmax (greedy), maximal 48 neue Tokens, Stop bei EOS. Alle 16 Fortsetzungen erreichten hier das 48-Token-Limit; Enden können mitten im Satz oder Wort liegen. Die Fortsetzungen sind Modellausgaben, keine geprüften Fakten und keine Wahrscheinlichkeitsverteilung über vollständige Antworten.

## Wahrscheinlichste nächste Tokens

| Land | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 | Masse der Top 10 |
| --- | --- | --- | --- | --- | --- | ---: |
| Australia | <code>" its"</code>: 68.59 % | <code>" having"</code>: 4.92 % | <code>" being"</code>: 4.27 % | <code>" the"</code>: 3.02 % | <code>" many"</code>: 1.75 % | 86.27 % |
| Argentina | <code>" its"</code>: 70.77 % | <code>" the"</code>: 2.64 % | <code>" being"</code>: 2.53 % | <code>" having"</code>: 2.27 % | <code>" producing"</code>: 1.71 % | 84.20 % |
| Egypt | <code>" its"</code>: 61.43 % | <code>" the"</code>: 6.07 % | <code>" being"</code>: 5.19 % | <code>" ancient"</code>: 2.58 % | <code>" having"</code>: 2.18 % | 81.30 % |
| Peru | <code>" its"</code>: 68.55 % | <code>" being"</code>: 4.92 % | <code>" the"</code>: 3.65 % | <code>" having"</code>: 2.09 % | <code>" producing"</code>: 0.90 % | 82.62 % |
| Japan | <code>" its"</code>: 67.85 % | <code>" being"</code>: 5.09 % | <code>" having"</code>: 2.87 % | <code>" the"</code>: 2.44 % | <code>" many"</code>: 1.72 % | 84.29 % |
| United States | <code>" its"</code>: 51.67 % | <code>" the"</code>: 5.96 % | <code>" being"</code>: 5.93 % | <code>" having"</code>: 3.57 % | <code>" many"</code>: 2.85 % | 75.08 % |
| China | <code>" its"</code>: 60.59 % | <code>" being"</code>: 7.26 % | <code>" producing"</code>: 6.18 % | <code>" having"</code>: 5.30 % | <code>" the"</code>: 3.91 % | 87.36 % |
| Brazil | <code>" its"</code>: 60.59 % | <code>" having"</code>: 5.25 % | <code>" being"</code>: 4.46 % | <code>" producing"</code>: 3.58 % | <code>" the"</code>: 3.41 % | 82.09 % |
| Venezuela | <code>" its"</code>: 63.73 % | <code>" being"</code>: 4.67 % | <code>" the"</code>: 4.54 % | <code>" having"</code>: 2.52 % | <code>" producing"</code>: 1.04 % | 78.99 % |
| India | <code>" its"</code>: 68.32 % | <code>" the"</code>: 4.53 % | <code>" being"</code>: 3.05 % | <code>" having"</code>: 2.30 % | <code>" a"</code>: 2.02 % | 84.64 % |
| Chile | <code>" its"</code>: 71.69 % | <code>" the"</code>: 3.66 % | <code>" being"</code>: 3.40 % | <code>" producing"</code>: 2.73 % | <code>" having"</code>: 2.70 % | 86.81 % |
| Saudi Arabia | <code>" its"</code>: 67.28 % | <code>" being"</code>: 5.85 % | <code>" the"</code>: 3.06 % | <code>" having"</code>: 1.79 % | <code>" producing"</code>: 1.41 % | 83.27 % |
| Germany | <code>" its"</code>: 67.46 % | <code>" being"</code>: 3.36 % | <code>" the"</code>: 3.34 % | <code>" having"</code>: 3.20 % | <code>" many"</code>: 2.21 % | 84.30 % |
| Spain | <code>" its"</code>: 74.15 % | <code>" being"</code>: 2.61 % | <code>" the"</code>: 2.29 % | <code>" having"</code>: 1.45 % | <code>" a"</code>: 1.34 % | 85.86 % |
| Russia | <code>" its"</code>: 71.02 % | <code>" being"</code>: 4.32 % | <code>" having"</code>: 3.94 % | <code>" producing"</code>: 3.44 % | <code>" the"</code>: 2.40 % | 88.77 % |
| South Korea | <code>" its"</code>: 66.81 % | <code>" being"</code>: 3.84 % | <code>" the"</code>: 3.53 % | <code>" having"</code>: 2.78 % | <code>" producing"</code>: 1.14 % | 81.02 % |

## Vollständige greedy Fortsetzungen bis zum 48-Token-Limit

Die Blöcke enthalten den Originalprompt plus die generierte Fortsetzung. Die Markierung des Token-Limits gehört nicht zur Modellausgabe.

### Australia

```text
Australia is known for its unique wildlife, and one of the most iconic animals is the kangaroo. These marsupials are not only fascinating to observe but also play a crucial role in the ecosystem. In this article, we will explore the different types of kang
```

Stopp: 48 neue Tokens, Token-Limit.

### Argentina

```text
Argentina is known for its rich cultural heritage, stunning landscapes, and delicious cuisine. However, one aspect of Argentine culture that often goes unnoticed is the country's unique approach to time management. In this article, we will explore the concept of "Argentine time"
```

Stopp: 48 neue Tokens, Token-Limit.

### Egypt

```text
Egypt is known for its ancient history, rich culture, and iconic landmarks like the pyramids and the Sphinx. The country is located in North Africa and is bordered by the Mediterranean Sea to the north, the Red Sea to the east, and Sudan and Libya to
```

Stopp: 48 neue Tokens, Token-Limit.

### Peru

```text
Peru is known for its rich cultural heritage, stunning landscapes, and diverse cuisine. However, one aspect of Peruvian culture that often goes unnoticed is its unique approach to timekeeping. In this article, we will explore the concept of "Peru Time" and
```

Stopp: 48 neue Tokens, Token-Limit.

### Japan

```text
Japan is known for its cherry blossoms, which are a symbol of the country's culture and beauty. The cherry blossom season typically occurs in the spring, and it is a time when many people in Japan go out to enjoy the flowers. However, the cherry
```

Stopp: 48 neue Tokens, Token-Limit.

### United States

```text
United States is known for its diverse culture, and one of the most prominent aspects of this diversity is the variety of cuisines available. From traditional American dishes to international flavors, the United States offers a wide range of culinary experiences. In this article, we will explore
```

Stopp: 48 neue Tokens, Token-Limit.

### China

```text
China is known for its rich cultural heritage and diverse traditions. One of the most iconic symbols of Chinese culture is the dragon, which is often associated with power, strength, and good fortune. In Chinese mythology, dragons are believed to control the weather and bring rain
```

Stopp: 48 neue Tokens, Token-Limit.

### Brazil

```text
Brazil is known for its rich cultural heritage, diverse landscapes, and vibrant cities. Here are some key facts about Brazil:

1. **Population**: Brazil is the most populous country in South America and the fifth most populous in the world, with over 21
```

Stopp: 48 neue Tokens, Token-Limit.

### Venezuela

```text
Venezuela is known for its oil reserves, but it also has a rich history of producing high-quality coffee. The country's coffee production dates back to the 18th century, and it has since become an important part of the country's economy and culture.

```

Stopp: 48 neue Tokens, Token-Limit.

### India

```text
India is known for its rich cultural heritage and diverse traditions. One of the most fascinating aspects of Indian culture is its cuisine, which is as diverse as the country itself. From the spicy curries of the south to the sweet desserts of the north, Indian food
```

Stopp: 48 neue Tokens, Token-Limit.

### Chile

```text
Chile is known for its long coastline, and the country's fishing industry is a major contributor to its economy. However, the fishing industry has faced challenges in recent years due to overfishing and environmental concerns. In this context, the Chilean government has implemented
```

Stopp: 48 neue Tokens, Token-Limit.

### Saudi Arabia

```text
Saudi Arabia is known for its oil reserves, but it also has a significant amount of natural gas reserves. The country has been investing in renewable energy sources, such as solar and wind power, to diversify its energy mix and reduce its carbon footprint. In addition,
```

Stopp: 48 neue Tokens, Token-Limit.

### Germany

```text
Germany is known for its high-quality education system, and the country has a long history of producing world-class scientists and researchers. The German education system is highly regarded for its emphasis on practical skills and hands-on learning, which is reflected in the country's strong engineering
```

Stopp: 48 neue Tokens, Token-Limit.

### Spain

```text
Spain is known for its rich history, vibrant culture, and stunning landscapes. From the bustling streets of Madrid to the sun-soaked beaches of the Costa del Sol, there is something for everyone in this diverse country. In this article, we will explore the top
```

Stopp: 48 neue Tokens, Token-Limit.

### Russia

```text
Russia is known for its vast natural resources, including oil, natural gas, and minerals. However, the country also has a rich cultural heritage and a diverse population. Can you provide a comprehensive analysis of the impact of Russia's natural resources on its economy and society
```

Stopp: 48 neue Tokens, Token-Limit.

### South Korea

```text
South Korea is known for its high-tech industries, and the country has a strong focus on innovation and technology. The country has a well-developed infrastructure, including a modern transportation system and a reliable internet connection. South Korea is also known for its high-quality education system
```

Stopp: 48 neue Tokens, Token-Limit.

## Artefakt-Kontrolle im normalen Forward-Pass

Die Ränge beziehen sich jeweils auf das erste nächste Token unter dem neutralen Prompt, nicht auf die Hauptstadt-ICL-Abfrage.

| Land | Rang `-Semit` | P(`-Semit`) | Rang `(Initialized` | P(`(Initialized`) |
| --- | ---: | ---: | ---: | ---: |
| Australia | 151902 | 6.35684e-16 | 151733 | 6.87317e-15 |
| Argentina | 151799 | 4.56073e-15 | 151738 | 7.15525e-15 |
| Egypt | 151407 | 4.66936e-14 | 151408 | 4.65637e-14 |
| Peru | 151331 | 3.32626e-14 | 151205 | 3.99895e-14 |
| Japan | 151902 | 4.55609e-16 | 151434 | 2.15816e-14 |
| United States | 151826 | 2.10551e-14 | 151768 | 4.30796e-14 |
| China | 151917 | 4.61922e-16 | 151565 | 3.98915e-14 |
| Brazil | 151879 | 3.12786e-15 | 151357 | 5.42223e-14 |
| Venezuela | 151834 | 7.99512e-15 | 151771 | 1.45297e-14 |
| India | 151799 | 6.72718e-15 | 151201 | 4.28592e-14 |
| Chile | 151370 | 2.56143e-14 | 151739 | 7.71621e-15 |
| Saudi Arabia | 151814 | 4.81652e-15 | 151513 | 1.92657e-14 |
| Germany | 151837 | 4.87699e-15 | 151568 | 2.11568e-14 |
| Spain | 151737 | 6.71189e-15 | 151206 | 3.76418e-14 |
| Russia | 151851 | 1.56935e-15 | 151808 | 2.95368e-15 |
| South Korea | 151872 | 1.45469e-15 | 151227 | 4.08168e-14 |

## Reproduktion

```sh
cd code
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu python ../paper/repro/qwen_token_readout.py
uv run python ../paper/repro/report_qwen_bug_diagnosis.py
```

[GPU-Skript](../../paper/repro/qwen_token_readout.py) · [Export-/Berichtsskript](../../paper/repro/report_qwen_bug_diagnosis.py). Modellrevision, Bibliotheksversionen und Source-Hashes sind im JSON dokumentiert.
