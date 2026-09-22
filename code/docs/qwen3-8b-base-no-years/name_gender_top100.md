# Qwen3 8B Base: top 100 J-lens predictions for `name_gender`

**100 ranked predictions for each of all 11 saved examples (1,100 entries).** Sofia occurs twice because the saved evaluation includes that name twice. Each table contains vocabulary tokens ranked by the differential J-lens, not a generated sequence.

[CSV: all 1,100 predictions](name_gender_top100.csv) · [JSON: predictions, scores, and provenance](name_gender_top100.json) · [Top-five summary and interpretation](name_gender_predictions.md)

## Examples

| Row | Name / predictions | Model target | Target rank (0-based) |
| ---: | --- | --- | ---: |
| 1 | [Tom — top 100](#row-1-tom) | `man` | 60,513 |
| 2 | [Mia — top 100](#row-2-mia) | `woman` | 108,458 |
| 3 | [Sofia — top 100](#row-3-sofia) | `woman` | 9,365 |
| 4 | [Connor — top 100](#row-4-connor) | `man` | 101,074 |
| 5 | [Evan — top 100](#row-5-evan) | `man` | 43,448 |
| 6 | [Sofia — top 100](#row-6-sofia) | `woman` | 9,365 |
| 7 | [Natalie — top 100](#row-7-natalie) | `woman` | 22,301 |
| 8 | [David — top 100](#row-8-david) | `man` | 11,338 |
| 9 | [Dylan — top 100](#row-9-dylan) | `man` | 110,309 |
| 10 | [Emily — top 100](#row-10-emily) | `woman` | 98,417 |
| 11 | [Michael — top 100](#row-11-michael) | `man` | 30,483 |

The top-100 lists were newly exported using the same Qwen3 8B Base checkpoint (`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`), newline prompt `\n{name} is known for`, fixed state 19, float32/eager attention, batch size 8, examples, and references as the plotted run. **The complete rerun measurement matches the previous saved measurement exactly**, including every target, target rank, top-five prediction, and aggregate. The existing plot and correlations are unchanged.

Prediction ranks below are **1-based** (1 = highest scoring token); target ranks above retain the original **0-based** convention. Tokens are displayed with surrounding whitespace stripped, as in the original table. Token IDs distinguish separate vocabulary tokens that look the same. The CSV (`token_json`, a JSON-escaped string) and JSON retain the exact decoded strings, including whitespace, and raw readout scores (not probabilities). None of the 11 model targets is in its corresponding top 100.

## Top-100 tables

Expand a name to see its complete ranked list. The first table is open by default.

### Row 1: Tom

<details open>
<summary>Tom: predictions 1–100 · target man · target rank 60,513 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` 萝卜 ` | 103438 |
| 2 | ` 法国 ` | 104328 |
| 3 | ` ** ` | 56177 |
| 4 | ` 苹果 ` | 104167 |
| 5 | ` ** ` | 334 |
| 6 | ` Tommy ` | 39039 |
| 7 | ` tomato ` | 41020 |
| 8 | ` :** ` | 66963 |
| 9 | ` 湘 ` | 101351 |
| 10 | ` ** ` | 1019 |
| 11 | ` 汤 ` | 102022 |
| 12 | ` **( ` | 83725 |
| 13 | ` 姜 ` | 102282 |
| 14 | ` lobster ` | 79715 |
| 15 | ` 番茄 ` | 113400 |
| 16 | ` 西红柿 ` | 117700 |
| 17 | ` 酱 ` | 102675 |
| 18 | ` apples ` | 40676 |
| 19 | ` 糖 ` | 100443 |
| 20 | ` )** ` | 32295 |
| 21 | ` Tom ` | 24732 |
| 22 | ` 唐 ` | 100181 |
| 23 | ` � ` | 98313 |
| 24 | ` 计算 ` | 100768 |
| 25 | ` 鲫 ` | 119492 |
| 26 | ` � ` | 59053 |
| 27 | ` tomatoes ` | 40513 |
| 28 | ` 蒋 ` | 102532 |
| 29 | ` 猪 ` | 100761 |
| 30 | ` 湖南省 ` | 106470 |
| 31 | ` 定制 ` | 104790 |
| 32 | ` ￥ ` | 72198 |
| 33 | ` 煲 ` | 114728 |
| 34 | ` 意大利 ` | 105101 |
| 35 | ` (** ` | 21798 |
| 36 | ` ：< ` | 90737 |
| 37 | ` fries ` | 51931 |
| 38 | ` 蟹 ` | 103111 |
| 39 | ` **, ` | 97219 |
| 40 | ` 茄 ` | 101833 |
| 41 | ` ** ` | 59479 |
| 42 | ` 爱吃 ` | 116489 |
| 43 | ` 专卖 ` | 115681 |
| 44 | ` ** ` | 3070 |
| 45 | ` 专 ` | 95411 |
| 46 | ` 松 ` | 100180 |
| 47 | ` 渔 ` | 100938 |
| 48 | ` **) ` | 41942 |
| 49 | ` 喜欢吃 ` | 115844 |
| 50 | ` 进货 ` | 117124 |
| 51 | ` 家 ` | 45629 |
| 52 | ` ^\ ` | 24884 |
| 53 | ` french ` | 41193 |
| 54 | ` 牛 ` | 100664 |
| 55 | ` 菜 ` | 99800 |
| 56 | ` 。\ ` | 66898 |
| 57 | ` 津 ` | 100468 |
| 58 | ` \( ` | 44292 |
| 59 | ` 白菜 ` | 111822 |
| 60 | ` 肿 ` | 101561 |
| 61 | ` 钓 ` | 103429 |
| 62 | ` 胡萝卜 ` | 111761 |
| 63 | ` 堤 ` | 107617 |
| 64 | ` <: ` | 47926 |
| 65 | ` 和 ` | 58143 |
| 66 | ` 商店 ` | 106899 |
| 67 | ` 篱 ` | 119081 |
| 68 | ` 宋 ` | 73038 |
| 69 | ` 北京市 ` | 104554 |
| 70 | ` 煮 ` | 104783 |
| 71 | ` Dorothy ` | 72558 |
| 72 | ` 第 ` | 29437 |
| 73 | ` 袜 ` | 116940 |
| 74 | ` 咸 ` | 102805 |
| 75 | ` 荷兰 ` | 108111 |
| 76 | ` holland ` | 64664 |
| 77 | ` 谭 ` | 106577 |
| 78 | ` 不要 ` | 100148 |
| 79 | ` appet ` | 29068 |
| 80 | ` 售价 ` | 107249 |
| 81 | ` 推荐 ` | 101914 |
| 82 | ` crab ` | 59412 |
| 83 | ` apple ` | 23268 |
| 84 | ` Jerry ` | 28708 |
| 85 | ` 【 ` | 10904 |
| 86 | ` **) ` | 27876 |
| 87 | ` � ` | 58230 |
| 88 | ` 吃完 ` | 109327 |
| 89 | ` 一道 ` | 104511 |
| 90 | ` 。< ` | 89393 |
| 91 | ` 天津 ` | 102133 |
| 92 | ` 戚 ` | 103210 |
| 93 | ` 】【 ` | 19574 |
| 94 | ` $$ ` | 26107 |
| 95 | ` � ` | 33424 |
| 96 | ` $$ ` | 14085 |
| 97 | ` 佟 ` | 118392 |
| 98 | ` 狸 ` | 103702 |
| 99 | ` 慈 ` | 101530 |
| 100 | ` ,** ` | 69979 |

</details>

### Row 2: Mia

<details>
<summary>Mia: predictions 1–100 · target woman · target rank 108,458 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` herself ` | 11675 |
| 2 | ` her ` | 1059 |
| 3 | ` 她的 ` | 104007 |
| 4 | ` 让她 ` | 105655 |
| 5 | ` 她在 ` | 108333 |
| 6 | ` Mia ` | 60597 |
| 7 | ` 她 ` | 99517 |
| 8 | ` 但她 ` | 110204 |
| 9 | ` 告诉她 ` | 117154 |
| 10 | ` 给她 ` | 106967 |
| 11 | ` .her ` | 43772 |
| 12 | ` 对她 ` | 108663 |
| 13 | ` 成都市 ` | 109581 |
| 14 | ` .', ` | 97469 |
| 15 | ` 成都 ` | 104051 |
| 16 | ` 因为她 ` | 113057 |
| 17 | ` )** ` | 32295 |
| 18 | ` **) ` | 27876 |
| 19 | ` 她是 ` | 107852 |
| 20 | ` 妞 ` | 114049 |
| 21 | ` .') ` | 54028 |
| 22 | ` **/ ` | 88490 |
| 23 | ` \ ` | 21672 |
| 24 | ` babys ` | 70583 |
| 25 | ` .") ` | 63173 |
| 26 | ` ', ` | 5550 |
| 27 | ` .") ` | 38739 |
| 28 | ` .'); ` | 78254 |
| 29 | ` 把她 ` | 110284 |
| 30 | ` .", ` | 57374 |
| 31 | ` :** ` | 66963 |
| 32 | ` **) ` | 41942 |
| 33 | ` **, ` | 97219 |
| 34 | ` Maui ` | 92450 |
| 35 | ` ') ` | 7694 |
| 36 | ` 跟她 ` | 113647 |
| 37 | ` ') ` | 77009 |
| 38 | ` .'. ` | 35949 |
| 39 | ` \ ` | 38178 |
| 40 | ` hers ` | 10837 |
| 41 | ` Liam ` | 60642 |
| 42 | ` :') ` | 79895 |
| 43 | ` {}) ` | 28875 |
| 44 | ` .') ` | 48320 |
| 45 | ` ') ` | 30018 |
| 46 | ` mommy ` | 81140 |
| 47 | ` ） ` | 91278 |
| 48 | ` ,** ` | 69979 |
| 49 | ` Giovanni ` | 75852 |
| 50 | ` 。\ ` | 66898 |
| 51 | ` ", ` | 4723 |
| 52 | ` ') ` | 863 |
| 53 | ` .': ` | 87088 |
| 54 | ` 婴幼儿 ` | 114225 |
| 55 | ` Bruno ` | 51110 |
| 56 | ` )", ` | 86726 |
| 57 | ` Julien ` | 87642 |
| 58 | ` Babies ` | 92692 |
| 59 | ` .) ` | 6138 |
| 60 | ` Noah ` | 41984 |
| 61 | ` Chloe ` | 59370 |
| 62 | ` ") ` | 50841 |
| 63 | ` 蕊 ` | 114753 |
| 64 | ` 婴 ` | 100592 |
| 65 | ` 傣 ` | 120984 |
| 66 | ` '}, ` | 76327 |
| 67 | ` preschool ` | 60805 |
| 68 | ` ()) ` | 13116 |
| 69 | ` .*) ` | 96494 |
| 70 | ` :") ` | 90220 |
| 71 | ` \\ ` | 24984 |
| 72 | ` Rihanna ` | 95427 |
| 73 | ` \\" ` | 69947 |
| 74 | ` 婴儿 ` | 102833 |
| 75 | ` :", ` | 71863 |
| 76 | ` ):: ` | 62256 |
| 77 | ` (json ` | 9304 |
| 78 | ` ") ` | 6060 |
| 79 | ` "}, ` | 45426 |
| 80 | ` ."); ` | 27007 |
| 81 | ` ') ` | 16667 |
| 82 | ` ()) ` | 2140 |
| 83 | ` 郪 ` | 122613 |
| 84 | ` 绘本 ` | 118800 |
| 85 | ` 儋 ` | 121643 |
| 86 | ` 孩子们 ` | 104563 |
| 87 | ` Bieber ` | 76220 |
| 88 | ` 邛 ` | 119555 |
| 89 | ` Ava ` | 91657 |
| 90 | ` she ` | 1340 |
| 91 | ` .) ` | 9272 |
| 92 | ` ） ` | 7552 |
| 93 | ` Olivia ` | 59287 |
| 94 | ` .". ` | 36911 |
| 95 | ` .") ` | 31225 |
| 96 | ` ") ` | 30218 |
| 97 | ` ).. ` | 88512 |
| 98 | ` ." ` | 71634 |
| 99 | ` ") ` | 899 |
| 100 | ` 亲子 ` | 107963 |

</details>

### Row 3: Sofia

<details>
<summary>Sofia: predictions 1–100 · target woman · target rank 9,365 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` Sofia ` | 78563 |
| 2 | ` Bulgaria ` | 58719 |
| 3 | ` herself ` | 11675 |
| 4 | ` Elena ` | 68575 |
| 5 | ` her ` | 1059 |
| 6 | ` Albania ` | 94633 |
| 7 | ` Estonia ` | 68362 |
| 8 | ` 她的 ` | 104007 |
| 9 | ` Catalonia ` | 95852 |
| 10 | ` Bulgarian ` | 88624 |
| 11 | ` ** ` | 59479 |
| 12 | ` 但她 ` | 110204 |
| 13 | ` Cristina ` | 99114 |
| 14 | ` María ` | 82205 |
| 15 | ` ', ` | 5550 |
| 16 | ` Venezuel ` | 70814 |
| 17 | ` Romania ` | 46049 |
| 18 | ` ': ` | 18421 |
| 19 | ` Dimit ` | 87590 |
| 20 | ` ） ` | 91278 |
| 21 | ` Dmitry ` | 87131 |
| 22 | ` Kosovo ` | 79897 |
| 23 | ` '); ` | 6416 |
| 24 | ` Barcelona ` | 26937 |
| 25 | ` 把她 ` | 110284 |
| 26 | ` Serbia ` | 58714 |
| 27 | ` 给她 ` | 106967 |
| 28 | ` Portugal ` | 33311 |
| 29 | ` Madrid ` | 24081 |
| 30 | ` ') ` | 7694 |
| 31 | ` 她在 ` | 108333 |
| 32 | ` Prague ` | 67289 |
| 33 | ` Viktor ` | 76016 |
| 34 | ` её ` | 134709 |
| 35 | ` '): ` | 60038 |
| 36 | ` Croatia ` | 57211 |
| 37 | ` ')) ` | 40317 |
| 38 | ` ? ` | 45349 |
| 39 | ` Milan ` | 38755 |
| 40 | ` '); ` | 25447 |
| 41 | ` '), ` | 22607 |
| 42 | ` Slovenia ` | 78217 |
| 43 | ` Kiev ` | 61625 |
| 44 | ` Spain ` | 74320 |
| 45 | ` 她 ` | 99517 |
| 46 | ` Mikhail ` | 91451 |
| 47 | ` ' ` | 9739 |
| 48 | ` '}, ` | 76327 |
| 49 | ` Armenia ` | 71530 |
| 50 | ` Sophia ` | 70371 |
| 51 | ` Sergey ` | 73429 |
| 52 | ` Cristiano ` | 99115 |
| 53 | ` Lisbon ` | 80701 |
| 54 | ` Valencia ` | 60249 |
| 55 | ` Putin ` | 67686 |
| 56 | ` Eston ` | 53323 |
| 57 | ` Alexandre ` | 89054 |
| 58 | ` Maria ` | 68556 |
| 59 | ` 因为她 ` | 113057 |
| 60 | ` 对她 ` | 108663 |
| 61 | ` Macedonia ` | 76409 |
| 62 | ` 葡萄牙 ` | 112334 |
| 63 | ` 让她 ` | 105655 |
| 64 | ` '], ` | 30390 |
| 65 | ` 基辅 ` | 117572 |
| 66 | ` Azerbaijan ` | 73477 |
| 67 | ` Ukraine ` | 18753 |
| 68 | ` Crimea ` | 60711 |
| 69 | ` Alejandro ` | 84919 |
| 70 | ` } ` | 1771 |
| 71 | ` Sevilla ` | 94894 |
| 72 | ` Sergei ` | 83998 |
| 73 | ` .', ` | 97469 |
| 74 | ` .'); ` | 78254 |
| 75 | ` '; ` | 7569 |
| 76 | ` Alberto ` | 64681 |
| 77 | ` Spain ` | 17689 |
| 78 | ` 。 ` | 37265 |
| 79 | ` ']) ` | 39573 |
| 80 | ` Porto ` | 69698 |
| 81 | ` Maria ` | 23016 |
| 82 | ` Lorenzo ` | 73400 |
| 83 | ` Luca ` | 83178 |
| 84 | ` Sergio ` | 65194 |
| 85 | ` she ` | 1340 |
| 86 | ` Pavel ` | 78269 |
| 87 | ` Carlo ` | 57770 |
| 88 | ` Bulg ` | 41250 |
| 89 | ` ?', ` | 41920 |
| 90 | ` Sonia ` | 89391 |
| 91 | ` Aleks ` | 89578 |
| 92 | ` ее ` | 128992 |
| 93 | ` 跟她 ` | 113647 |
| 94 | ` '] ` | 28673 |
| 95 | ` Monaco ` | 61551 |
| 96 | ` Istanbul ` | 53368 |
| 97 | ` Ukrain ` | 30604 |
| 98 | ` 米兰 ` | 109049 |
| 99 | ` Roberto ` | 54083 |
| 100 | ` São ` | 47690 |

</details>

### Row 4: Connor

<details>
<summary>Connor: predictions 1–100 · target man · target rank 101,074 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` Connor ` | 57180 |
| 2 | ` Conor ` | 84115 |
| 3 | ` Connor ` | 56887 |
| 4 | ` Brendan ` | 55843 |
| 5 | ` Quinn ` | 40523 |
| 6 | ` Dillon ` | 82566 |
| 7 | ` Liam ` | 60642 |
| 8 | ` Newfoundland ` | 88296 |
| 9 | ` Jake ` | 94236 |
| 10 | ` Jake ` | 32072 |
| 11 | ` Sharks ` | 70823 |
| 12 | ` Aussie ` | 86463 |
| 13 | ` Bannon ` | 54165 |
| 14 | ` Canucks ` | 80502 |
| 15 | ` Shane ` | 49975 |
| 16 | ` Joey ` | 54735 |
| 17 | ` Devon ` | 59334 |
| 18 | ` Fallon ` | 87068 |
| 19 | ` Irish ` | 17623 |
| 20 | ` Corey ` | 54652 |
| 21 | ` Cait ` | 75332 |
| 22 | ` Seahawks ` | 44926 |
| 23 | ` Kirby ` | 57711 |
| 24 | ` Shaun ` | 76271 |
| 25 | ` Australian ` | 13369 |
| 26 | ` Derek ` | 42323 |
| 27 | ` Belfast ` | 54239 |
| 28 | ` Shelby ` | 69408 |
| 29 | ` Cassidy ` | 91797 |
| 30 | ` Brandon ` | 31766 |
| 31 | ` Swagger ` | 79631 |
| 32 | ` Nolan ` | 61632 |
| 33 | ` Australian ` | 74241 |
| 34 | ` Shawn ` | 55537 |
| 35 | ` Craig ` | 28417 |
| 36 | ` Courtney ` | 70682 |
| 37 | ` Sawyer ` | 91119 |
| 38 | ` Aaron ` | 25673 |
| 39 | ` Derrick ` | 72089 |
| 40 | ` Josh ` | 18246 |
| 41 | ` Donovan ` | 70700 |
| 42 | ` Doyle ` | 65209 |
| 43 | ` Caleb ` | 83062 |
| 44 | ` ustralian ` | 57149 |
| 45 | ` Cameron ` | 26426 |
| 46 | ` Conway ` | 59474 |
| 47 | ` hockey ` | 26953 |
| 48 | ` Tanner ` | 89796 |
| 49 | ` Wolverine ` | 83179 |
| 50 | ` Reef ` | 75936 |
| 51 | ` Kavanaugh ` | 37183 |
| 52 | ` Brennan ` | 58750 |
| 53 | ` Dawson ` | 66212 |
| 54 | ` Sean ` | 24995 |
| 55 | ` Slater ` | 97003 |
| 56 | ` .# ` | 43326 |
| 57 | ` Brandon ` | 93757 |
| 58 | ` Kyle ` | 30259 |
| 59 | ` Burke ` | 49623 |
| 60 | ` Kyle ` | 91180 |
| 61 | ` Bears ` | 35968 |
| 62 | ` Jared ` | 43228 |
| 63 | ` Scott ` | 9815 |
| 64 | ` Islanders ` | 78453 |
| 65 | ` Preston ` | 62225 |
| 66 | ` Frankie ` | 93445 |
| 67 | ` Crawford ` | 53176 |
| 68 | ` Kodi ` | 81312 |
| 69 | ` Sheffield ` | 60025 |
| 70 | ` surfing ` | 57740 |
| 71 | ` Wyatt ` | 87462 |
| 72 | ` 獒 ` | 121347 |
| 73 | ` Owen ` | 46709 |
| 74 | ` 谷爱凌 ` | 116087 |
| 75 | ` Winnipeg ` | 51882 |
| 76 | ` Scott ` | 38220 |
| 77 | ` Drake ` | 41765 |
| 78 | ` Shark ` | 60436 |
| 79 | ` Overwatch ` | 72338 |
| 80 | ` 爱尔兰 ` | 116517 |
| 81 | ` 澳大 ` | 105578 |
| 82 | ` Emily ` | 84048 |
| 83 | ` Gavin ` | 63395 |
| 84 | ` Ryder ` | 88023 |
| 85 | ` Jenna ` | 87254 |
| 86 | ` .json ` | 4323 |
| 87 | ` Megan ` | 54529 |
| 88 | ` Cornwall ` | 75950 |
| 89 | ` McGregor ` | 68016 |
| 90 | ` Blackhawks ` | 92392 |
| 91 | ` Rory ` | 82321 |
| 92 | ` McConnell ` | 46683 |
| 93 | ` Tristan ` | 96590 |
| 94 | ` Ashley ` | 37118 |
| 95 | ` Meghan ` | 79034 |
| 96 | ` lobster ` | 79715 |
| 97 | ` Jason ` | 18477 |
| 98 | ` Fraser ` | 55130 |
| 99 | ` Loch ` | 82703 |
| 100 | ` Drew ` | 40108 |

</details>

### Row 5: Evan

<details>
<summary>Evan: predictions 1–100 · target man · target rank 43,448 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` Vermont ` | 34639 |
| 2 | ` indie ` | 43478 |
| 3 | ` Evan ` | 43943 |
| 4 | ` Winnipeg ` | 51882 |
| 5 | ` surfing ` | 57740 |
| 6 | ` Vancouver ` | 22575 |
| 7 | ` Scottish ` | 26096 |
| 8 | ` Brendan ` | 55843 |
| 9 | ` vegan ` | 29149 |
| 10 | ` Newfoundland ` | 88296 |
| 11 | ` Asheville ` | 96538 |
| 12 | ` Surf ` | 64097 |
| 13 | ` shoreline ` | 98064 |
| 14 | ` Ethan ` | 62164 |
| 15 | ` Blackhawks ` | 92392 |
| 16 | ` Devon ` | 59334 |
| 17 | ` 湾区 ` | 103146 |
| 18 | ` Maui ` | 92450 |
| 19 | ` Brighton ` | 57419 |
| 20 | ` Heather ` | 46263 |
| 21 | ` surf ` | 20269 |
| 22 | ` Toronto ` | 14632 |
| 23 | ` Newport ` | 58800 |
| 24 | ` Hawaiian ` | 58003 |
| 25 | ` Seahawks ` | 44926 |
| 26 | ` Shopify ` | 95025 |
| 27 | ` Canadian ` | 11888 |
| 28 | ` Aussie ` | 86463 |
| 29 | ` Reef ` | 75936 |
| 30 | ` Denver ` | 22117 |
| 31 | ` Welsh ` | 45781 |
| 32 | ` skate ` | 42778 |
| 33 | ` Seattle ` | 16355 |
| 34 | ` Scotland ` | 19085 |
| 35 | ` Shawn ` | 55537 |
| 36 | ` Manit ` | 53235 |
| 37 | ` Boulder ` | 55643 |
| 38 | ` tattoo ` | 31794 |
| 39 | ` padd ` | 53169 |
| 40 | ` Kendrick ` | 87613 |
| 41 | ` Cleveland ` | 23475 |
| 42 | ` hoodie ` | 96301 |
| 43 | ` geek ` | 55639 |
| 44 | ` Irish ` | 17623 |
| 45 | ` Milwaukee ` | 35818 |
| 46 | ` Kanye ` | 58717 |
| 47 | ` Sheffield ` | 60025 |
| 48 | ` Edmonton ` | 45287 |
| 49 | ` Cornwall ` | 75950 |
| 50 | ` Scott ` | 9815 |
| 51 | ` amps ` | 79962 |
| 52 | ` streaming ` | 16842 |
| 53 | ` Kickstarter ` | 39980 |
| 54 | ` Brandon ` | 31766 |
| 55 | ` Canucks ` | 80502 |
| 56 | ` Leslie ` | 53683 |
| 57 | ` Colorado ` | 15383 |
| 58 | ` Auckland ` | 53246 |
| 59 | ` beach ` | 11321 |
| 60 | ` Fairfax ` | 76536 |
| 61 | ` Craig ` | 28417 |
| 62 | ` Australian ` | 13369 |
| 63 | ` Drake ` | 41765 |
| 64 | ` reef ` | 70045 |
| 65 | ` Dakota ` | 27872 |
| 66 | ` Seth ` | 44218 |
| 67 | ` Highland ` | 64087 |
| 68 | ` Preston ` | 62225 |
| 69 | ` glacier ` | 93767 |
| 70 | ` Jared ` | 43228 |
| 71 | ` Icelandic ` | 99148 |
| 72 | ` Erin ` | 55814 |
| 73 | ` Beach ` | 12721 |
| 74 | ` Rocky ` | 47541 |
| 75 | ` Brooklyn ` | 25748 |
| 76 | ` Sharks ` | 70823 |
| 77 | ` Appalachian ` | 97927 |
| 78 | ` Hawaii ` | 27521 |
| 79 | ` Derek ` | 42323 |
| 80 | ` Wisconsin ` | 20441 |
| 81 | ` Chicago ` | 10557 |
| 82 | ` Wyoming ` | 48732 |
| 83 | ` BuzzFeed ` | 71822 |
| 84 | ` Minnesota ` | 18925 |
| 85 | ` Canadians ` | 35162 |
| 86 | ` Leeds ` | 51529 |
| 87 | ` Honolulu ` | 81540 |
| 88 | ` Katie ` | 44463 |
| 89 | ` hockey ` | 26953 |
| 90 | ` Ontario ` | 20551 |
| 91 | ` Swansea ` | 77803 |
| 92 | ` canadian ` | 87598 |
| 93 | ` Tasmania ` | 92428 |
| 94 | ` Lindsay ` | 62784 |
| 95 | ` Manitoba ` | 63240 |
| 96 | ` Ohio ` | 14352 |
| 97 | ` hiking ` | 37364 |
| 98 | ` Scott ` | 38220 |
| 99 | ` Connecticut ` | 30361 |
| 100 | ` Prairie ` | 70949 |

</details>

### Row 6: Sofia

<details>
<summary>Sofia: predictions 1–100 · target woman · target rank 9,365 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` Sofia ` | 78563 |
| 2 | ` Bulgaria ` | 58719 |
| 3 | ` herself ` | 11675 |
| 4 | ` Elena ` | 68575 |
| 5 | ` her ` | 1059 |
| 6 | ` Albania ` | 94633 |
| 7 | ` Estonia ` | 68362 |
| 8 | ` 她的 ` | 104007 |
| 9 | ` Catalonia ` | 95852 |
| 10 | ` Bulgarian ` | 88624 |
| 11 | ` ** ` | 59479 |
| 12 | ` 但她 ` | 110204 |
| 13 | ` Cristina ` | 99114 |
| 14 | ` María ` | 82205 |
| 15 | ` ', ` | 5550 |
| 16 | ` Venezuel ` | 70814 |
| 17 | ` Romania ` | 46049 |
| 18 | ` ': ` | 18421 |
| 19 | ` Dimit ` | 87590 |
| 20 | ` ） ` | 91278 |
| 21 | ` Dmitry ` | 87131 |
| 22 | ` Kosovo ` | 79897 |
| 23 | ` '); ` | 6416 |
| 24 | ` Barcelona ` | 26937 |
| 25 | ` 把她 ` | 110284 |
| 26 | ` Serbia ` | 58714 |
| 27 | ` 给她 ` | 106967 |
| 28 | ` Portugal ` | 33311 |
| 29 | ` Madrid ` | 24081 |
| 30 | ` ') ` | 7694 |
| 31 | ` 她在 ` | 108333 |
| 32 | ` Prague ` | 67289 |
| 33 | ` Viktor ` | 76016 |
| 34 | ` её ` | 134709 |
| 35 | ` '): ` | 60038 |
| 36 | ` Croatia ` | 57211 |
| 37 | ` ')) ` | 40317 |
| 38 | ` ? ` | 45349 |
| 39 | ` Milan ` | 38755 |
| 40 | ` '); ` | 25447 |
| 41 | ` '), ` | 22607 |
| 42 | ` Slovenia ` | 78217 |
| 43 | ` Kiev ` | 61625 |
| 44 | ` Spain ` | 74320 |
| 45 | ` 她 ` | 99517 |
| 46 | ` Mikhail ` | 91451 |
| 47 | ` ' ` | 9739 |
| 48 | ` '}, ` | 76327 |
| 49 | ` Armenia ` | 71530 |
| 50 | ` Sophia ` | 70371 |
| 51 | ` Sergey ` | 73429 |
| 52 | ` Cristiano ` | 99115 |
| 53 | ` Lisbon ` | 80701 |
| 54 | ` Valencia ` | 60249 |
| 55 | ` Putin ` | 67686 |
| 56 | ` Eston ` | 53323 |
| 57 | ` Alexandre ` | 89054 |
| 58 | ` Maria ` | 68556 |
| 59 | ` 因为她 ` | 113057 |
| 60 | ` 对她 ` | 108663 |
| 61 | ` Macedonia ` | 76409 |
| 62 | ` 葡萄牙 ` | 112334 |
| 63 | ` 让她 ` | 105655 |
| 64 | ` '], ` | 30390 |
| 65 | ` 基辅 ` | 117572 |
| 66 | ` Azerbaijan ` | 73477 |
| 67 | ` Ukraine ` | 18753 |
| 68 | ` Crimea ` | 60711 |
| 69 | ` Alejandro ` | 84919 |
| 70 | ` } ` | 1771 |
| 71 | ` Sevilla ` | 94894 |
| 72 | ` Sergei ` | 83998 |
| 73 | ` .', ` | 97469 |
| 74 | ` .'); ` | 78254 |
| 75 | ` '; ` | 7569 |
| 76 | ` Alberto ` | 64681 |
| 77 | ` Spain ` | 17689 |
| 78 | ` 。 ` | 37265 |
| 79 | ` ']) ` | 39573 |
| 80 | ` Porto ` | 69698 |
| 81 | ` Maria ` | 23016 |
| 82 | ` Lorenzo ` | 73400 |
| 83 | ` Luca ` | 83178 |
| 84 | ` Sergio ` | 65194 |
| 85 | ` she ` | 1340 |
| 86 | ` Pavel ` | 78269 |
| 87 | ` Carlo ` | 57770 |
| 88 | ` Bulg ` | 41250 |
| 89 | ` ?', ` | 41920 |
| 90 | ` Sonia ` | 89391 |
| 91 | ` Aleks ` | 89578 |
| 92 | ` ее ` | 128992 |
| 93 | ` 跟她 ` | 113647 |
| 94 | ` '] ` | 28673 |
| 95 | ` Monaco ` | 61551 |
| 96 | ` Istanbul ` | 53368 |
| 97 | ` Ukrain ` | 30604 |
| 98 | ` 米兰 ` | 109049 |
| 99 | ` Roberto ` | 54083 |
| 100 | ` São ` | 47690 |

</details>

### Row 7: Natalie

<details>
<summary>Natalie: predictions 1–100 · target woman · target rank 22,301 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` herself ` | 11675 |
| 2 | ` her ` | 1059 |
| 3 | ` Natalie ` | 63020 |
| 4 | ` Hawaiian ` | 58003 |
| 5 | ` Newfoundland ` | 88296 |
| 6 | ` Canadians ` | 35162 |
| 7 | ` she ` | 1340 |
| 8 | ` Vermont ` | 34639 |
| 9 | ` Katie ` | 44463 |
| 10 | ` surfing ` | 57740 |
| 11 | ` Surf ` | 64097 |
| 12 | ` Devon ` | 59334 |
| 13 | ` Hawaii ` | 27521 |
| 14 | ` Ecuador ` | 53991 |
| 15 | ` Nashville ` | 36540 |
| 16 | ` Edmonton ` | 45287 |
| 17 | ` Quebec ` | 34095 |
| 18 | ` Toronto ` | 14632 |
| 19 | ` surf ` | 20269 |
| 20 | ` Canadian ` | 11888 |
| 21 | ` Québec ` | 79188 |
| 22 | ` hers ` | 10837 |
| 23 | ` Australian ` | 13369 |
| 24 | ` Lindsay ` | 62784 |
| 25 | ` ranch ` | 38061 |
| 26 | ` Ontario ` | 20551 |
| 27 | ` Vancouver ` | 22575 |
| 28 | ` Australians ` | 55837 |
| 29 | ` Alberta ` | 32554 |
| 30 | ` 但她 ` | 110204 |
| 31 | ` Honolulu ` | 81540 |
| 32 | ` beach ` | 11321 |
| 33 | ` Jenner ` | 76064 |
| 34 | ` Maui ` | 92450 |
| 35 | ` reef ` | 70045 |
| 36 | ` 她的 ` | 104007 |
| 37 | ` Colorado ` | 15383 |
| 38 | ` NSW ` | 37908 |
| 39 | ` Lauren ` | 42360 |
| 40 | ` Kylie ` | 97164 |
| 41 | ` Monterey ` | 97395 |
| 42 | ` Reef ` | 75936 |
| 43 | ` Alison ` | 66363 |
| 44 | ` Danielle ` | 71616 |
| 45 | ` Belize ` | 97004 |
| 46 | ` Brighton ` | 57419 |
| 47 | ` farmhouse ` | 82308 |
| 48 | ` Hawai ` | 21437 |
| 49 | ` Rihanna ` | 95427 |
| 50 | ` Lana ` | 94787 |
| 51 | ` 把她 ` | 110284 |
| 52 | ` cottage ` | 45622 |
| 53 | ` Aussie ` | 86463 |
| 54 | ` Chanel ` | 90372 |
| 55 | ` Nicar ` | 66038 |
| 56 | ` padd ` | 53169 |
| 57 | ` glam ` | 41012 |
| 58 | ` Stacy ` | 97325 |
| 59 | ` Slater ` | 97003 |
| 60 | ` Sharon ` | 51852 |
| 61 | ` 她在 ` | 108333 |
| 62 | ` Jennifer ` | 29650 |
| 63 | ` MTV ` | 61099 |
| 64 | ` lobster ` | 79715 |
| 65 | ` Courtney ` | 70682 |
| 66 | ` Nicole ` | 44030 |
| 67 | ` shoreline ` | 98064 |
| 68 | ` Melanie ` | 84250 |
| 69 | ` Heidi ` | 81836 |
| 70 | ` beaches ` | 34809 |
| 71 | ` Hait ` | 42402 |
| 72 | ` Kardashian ` | 68007 |
| 73 | ` Tiffany ` | 67473 |
| 74 | ` Liverpool ` | 24688 |
| 75 | ` Whitney ` | 66097 |
| 76 | ` Winnipeg ` | 51882 |
| 77 | ` Auckland ` | 53246 |
| 78 | ` 对她 ` | 108663 |
| 79 | ` She ` | 2932 |
| 80 | ` Vogue ` | 89972 |
| 81 | ` Heather ` | 46263 |
| 82 | ` Newport ` | 58800 |
| 83 | ` Brazilian ` | 34983 |
| 84 | ` Texans ` | 52691 |
| 85 | ` Portland ` | 23077 |
| 86 | ` Scottish ` | 26096 |
| 87 | ` Dairy ` | 78022 |
| 88 | ` Swiss ` | 29691 |
| 89 | ` Sydney ` | 21273 |
| 90 | ` Sandra ` | 55686 |
| 91 | ` vodka ` | 75424 |
| 92 | ` Tasmania ` | 92428 |
| 93 | ` boutique ` | 52085 |
| 94 | ` Surrey ` | 66964 |
| 95 | ` Jamaica ` | 56175 |
| 96 | ` Oregon ` | 18784 |
| 97 | ` Richie ` | 98974 |
| 98 | ` Suffolk ` | 80783 |
| 99 | ` Nikki ` | 69107 |
| 100 | ` Belgian ` | 48062 |

</details>

### Row 8: David

<details>
<summary>David: predictions 1–100 · target man · target rank 11,338 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` french ` | 41193 |
| 2 | ` kosher ` | 97640 |
| 3 | ` Hawaiian ` | 58003 |
| 4 | ` Argentine ` | 81722 |
| 5 | ` Judaism ` | 63183 |
| 6 | ` jewish ` | 88138 |
| 7 | ` Jewish ` | 13350 |
| 8 | ` French ` | 8585 |
| 9 | ` Chile ` | 33000 |
| 10 | ` Saul ` | 67524 |
| 11 | ` Jews ` | 16987 |
| 12 | ` Hawaii ` | 27521 |
| 13 | ` jewellery ` | 62391 |
| 14 | ` holland ` | 64664 |
| 15 | ` brazil ` | 96646 |
| 16 | ` Ecuador ` | 53991 |
| 17 | ` Hawai ` | 21437 |
| 18 | ` Lebanese ` | 68245 |
| 19 | ` Holocaust ` | 43098 |
| 20 | ` Swiss ` | 29691 |
| 21 | ` Danish ` | 43680 |
| 22 | ` Adolf ` | 89485 |
| 23 | ` France ` | 9625 |
| 24 | ` Auschwitz ` | 92290 |
| 25 | ` Brazilian ` | 34983 |
| 26 | ` Dominican ` | 66013 |
| 27 | ` Belgian ` | 48062 |
| 28 | ` .... ` | 77828 |
| 29 | ` Portuguese ` | 42188 |
| 30 | ` compuls ` | 47106 |
| 31 | ` Yorkshire ` | 50227 |
| 32 | ` seaside ` | 92776 |
| 33 | ` Jew ` | 7958 |
| 34 | ` beach ` | 11321 |
| 35 | ` Argentina ` | 31064 |
| 36 | ` Moroccan ` | 93967 |
| 37 | ` martin ` | 94916 |
| 38 | ` jew ` | 16868 |
| 39 | ` spanish ` | 77032 |
| 40 | ` catholic ` | 82091 |
| 41 | ` seafood ` | 52305 |
| 42 | ` … ` | 89478 |
| 43 | ` French ` | 43197 |
| 44 | ` Cornwall ` | 75950 |
| 45 | ` seaw ` | 66229 |
| 46 | ` Puerto ` | 30219 |
| 47 | ` argent ` | 88144 |
| 48 | ` Spanish ` | 15154 |
| 49 | ` Mexican ` | 23279 |
| 50 | ` france ` | 47587 |
| 51 | ` Caribbean ` | 34274 |
| 52 | ` � ` | 126 |
| 53 | ` . ` | 16448 |
| 54 | ` marine ` | 28591 |
| 55 | ` 泰州 ` | 117053 |
| 56 | ` ... ` | 9338 |
| 57 | ` NYPD ` | 73155 |
| 58 | ` lobster ` | 79715 |
| 59 | ` σ ` | 47723 |
| 60 | ` cooking ` | 17233 |
| 61 | ` –and ` | 85923 |
| 62 | ` Catalonia ` | 95852 |
| 63 | ` Colombian ` | 83553 |
| 64 | ` Zimbabwe ` | 51220 |
| 65 | ` Cuban ` | 43710 |
| 66 | ` surfing ` | 57740 |
| 67 | ` Franco ` | 43843 |
| 68 | ` shore ` | 28426 |
| 69 | ` Albert ` | 17513 |
| 70 | ` ,... ` | 52388 |
| 71 | ` Hamburg ` | 50462 |
| 72 | ` Hebrew ` | 36266 |
| 73 | ` fusion ` | 36508 |
| 74 | ` bible ` | 60201 |
| 75 | ` feminism ` | 52010 |
| 76 | ` cuffs ` | 89469 |
| 77 | ` Rabbi ` | 70869 |
| 78 | ` harbour ` | 74642 |
| 79 | ` Portugal ` | 33311 |
| 80 | ` Israeli ` | 15907 |
| 81 | ` Icelandic ` | 99148 |
| 82 | ` coastline ` | 79844 |
| 83 | ` 阿根廷 ` | 109192 |
| 84 | ` Dutch ` | 23234 |
| 85 | ` elegance ` | 63443 |
| 86 | ` sea ` | 9396 |
| 87 | ` whale ` | 50019 |
| 88 | ` emphas ` | 20053 |
| 89 | ` farmhouse ` | 82308 |
| 90 | ` Torah ` | 71950 |
| 91 | ` Victorian ` | 43177 |
| 92 | ` ocean ` | 17951 |
| 93 | ` franc ` | 43479 |
| 94 | ` Iceland ` | 38396 |
| 95 | ` coastal ` | 34235 |
| 96 | ` Ricardo ` | 65950 |
| 97 | ` Spain ` | 17689 |
| 98 | ` Biblical ` | 67101 |
| 99 | ` island ` | 12922 |
| 100 | ` .. ` | 33947 |

</details>

### Row 9: Dylan

<details>
<summary>Dylan: predictions 1–100 · target man · target rank 110,309 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` Dylan ` | 43358 |
| 2 | ` Welsh ` | 45781 |
| 3 | ` Belfast ` | 54239 |
| 4 | ` Swansea ` | 77803 |
| 5 | ` Newfoundland ` | 88296 |
| 6 | ` Nashville ` | 36540 |
| 7 | ` Icelandic ` | 99148 |
| 8 | ` Cardiff ` | 60592 |
| 9 | ` Wales ` | 22931 |
| 10 | ` songwriter ` | 84657 |
| 11 | ` 爱尔兰 ` | 116517 |
| 12 | ` Irish ` | 17623 |
| 13 | ` Gael ` | 88638 |
| 14 | ` Winnipeg ` | 51882 |
| 15 | ` Tasmania ` | 92428 |
| 16 | ` 这首歌 ` | 117883 |
| 17 | ` indie ` | 43478 |
| 18 | ` .) ` | 9272 |
| 19 | ` 吉他 ` | 115789 |
| 20 | ` Scottish ` | 26096 |
| 21 | ` Celtic ` | 54664 |
| 22 | ` ...) ` | 61827 |
| 23 | ` .*) ` | 96494 |
| 24 | ` Rory ` | 82321 |
| 25 | ` Tolkien ` | 79303 |
| 26 | ` .) ` | 6138 |
| 27 | ` -song ` | 76640 |
| 28 | ` 湾区 ` | 103146 |
| 29 | ` Kanye ` | 58717 |
| 30 | ` 利物浦 ` | 115628 |
| 31 | ` 隰 ` | 121842 |
| 32 | ` .') ` | 48320 |
| 33 | ` Liam ` | 60642 |
| 34 | ` 威海 ` | 110582 |
| 35 | ` Liverpool ` | 24688 |
| 36 | ` Iceland ` | 38396 |
| 37 | ` 遵义 ` | 116365 |
| 38 | ` Beatles ` | 54857 |
| 39 | ` Tristan ` | 96590 |
| 40 | ` .") ` | 31225 |
| 41 | ` 摇滚 ` | 117278 |
| 42 | ` Brittany ` | 88154 |
| 43 | ` guitarist ` | 61640 |
| 44 | ` songs ` | 11674 |
| 45 | ` .") ` | 38739 |
| 46 | ` ..." ` | 47486 |
| 47 | ` Conor ` | 84115 |
| 48 | ` 乐队 ` | 111818 |
| 49 | ` .' ` | 21696 |
| 50 | ` ._ ` | 34773 |
| 51 | ` Brendan ` | 55843 |
| 52 | ` 澳大 ` | 105578 |
| 53 | ` ." ` | 2217 |
| 54 | ` Crosby ` | 75736 |
| 55 | ` Maine ` | 29790 |
| 56 | ` Yorkshire ` | 50227 |
| 57 | ` ylan ` | 36478 |
| 58 | ` .") ` | 63173 |
| 59 | ` Jonah ` | 93628 |
| 60 | ` blues ` | 43595 |
| 61 | ` 威尔 ` | 110840 |
| 62 | ` Norse ` | 93514 |
| 63 | ` 挪威 ` | 117451 |
| 64 | ` 峡 ` | 101011 |
| 65 | ` Asheville ` | 96538 |
| 66 | ` song ` | 5492 |
| 67 | ` .). ` | 74203 |
| 68 | ` Shawn ` | 55537 |
| 69 | ` 贵州省 ` | 109847 |
| 70 | ` 新西兰 ` | 109751 |
| 71 | ` Kylie ` | 97164 |
| 72 | ` .') ` | 54028 |
| 73 | ` .'] ` | 98758 |
| 74 | ` .\| ` | 56270 |
| 75 | ` spotify ` | 87790 |
| 76 | ` Samoa ` | 87020 |
| 77 | ` Dakota ` | 27872 |
| 78 | ` .wav ` | 29864 |
| 79 | ` Sheffield ` | 60025 |
| 80 | ` Nolan ` | 61632 |
| 81 | ` 专辑 ` | 109468 |
| 82 | ` guitar ` | 16986 |
| 83 | ` ."," ` | 47891 |
| 84 | ` .', ` | 97469 |
| 85 | ` Tennessee ` | 23219 |
| 86 | ` Kirby ` | 57711 |
| 87 | ` 南非 ` | 113453 |
| 88 | ` Bordeaux ` | 83957 |
| 89 | ` .’ ` | 35219 |
| 90 | ` 岬 ` | 119878 |
| 91 | ` .': ` | 87088 |
| 92 | ` ...) ` | 32574 |
| 93 | ` .). ` | 35334 |
| 94 | ` __) ` | 40725 |
| 95 | ` Auckland ` | 53246 |
| 96 | ` Leeds ` | 51529 |
| 97 | ` Scots ` | 74267 |
| 98 | ` 海边 ` | 113146 |
| 99 | ` Drake ` | 41765 |
| 100 | ` rocking ` | 66624 |

</details>

### Row 10: Emily

<details>
<summary>Emily: predictions 1–100 · target woman · target rank 98,417 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` Emily ` | 84048 |
| 2 | ` Emily ` | 34166 |
| 3 | ` 她的 ` | 104007 |
| 4 | ` her ` | 1059 |
| 5 | ` herself ` | 11675 |
| 6 | ` 她在 ` | 108333 |
| 7 | ` Danielle ` | 71616 |
| 8 | ` 但她 ` | 110204 |
| 9 | ` Alison ` | 66363 |
| 10 | ` 她 ` | 99517 |
| 11 | ` Eleanor ` | 89430 |
| 12 | ` Emma ` | 34935 |
| 13 | ` 宝玉 ` | 117148 |
| 14 | ` Emma ` | 89074 |
| 15 | ` Vermont ` | 34639 |
| 16 | ` Jenna ` | 87254 |
| 17 | ` Elizabeth ` | 75537 |
| 18 | ` Heather ` | 46263 |
| 19 | ` Jane ` | 21475 |
| 20 | ` 她是 ` | 107852 |
| 21 | ` Dickinson ` | 97237 |
| 22 | ` Heidi ` | 81836 |
| 23 | ` Jennifer ` | 71426 |
| 24 | ` Jane ` | 62502 |
| 25 | ` Claire ` | 42575 |
| 26 | ` Elizabeth ` | 20731 |
| 27 | ` Bennett ` | 38286 |
| 28 | ` Katie ` | 44463 |
| 29 | ` 对她 ` | 108663 |
| 30 | ` Rebecca ` | 44457 |
| 31 | ` Rachel ` | 83878 |
| 32 | ` Rachel ` | 31432 |
| 33 | ` etsy ` | 62627 |
| 34 | ` 雯 ` | 108956 |
| 35 | ` Jessica ` | 32367 |
| 36 | ` 广东省 ` | 104922 |
| 37 | ` 媛 ` | 109970 |
| 38 | ` Lindsay ` | 62784 |
| 39 | ` Sophie ` | 59310 |
| 40 | ` Laura ` | 29828 |
| 41 | ` Jessica ` | 78990 |
| 42 | ` Shelley ` | 90509 |
| 43 | ` 跟她 ` | 113647 |
| 44 | ` Jenny ` | 51079 |
| 45 | ` homeschool ` | 89819 |
| 46 | ` 芙 ` | 103639 |
| 47 | ` Megan ` | 54529 |
| 48 | ` Jennifer ` | 29650 |
| 49 | ` 把她 ` | 110284 |
| 50 | ` Stacy ` | 97325 |
| 51 | ` Nicole ` | 44030 |
| 52 | ` 常州 ` | 108923 |
| 53 | ` Laura ` | 76460 |
| 54 | ` Sarah ` | 20445 |
| 55 | ` Michelle ` | 82762 |
| 56 | ` Irene ` | 87213 |
| 57 | ` 江门 ` | 112927 |
| 58 | ` 给她 ` | 106967 |
| 59 | ` Erin ` | 55814 |
| 60 | ` Stephanie ` | 48143 |
| 61 | ` Watson ` | 31480 |
| 62 | ` Jake ` | 94236 |
| 63 | ` Isabel ` | 86786 |
| 64 | ` Cathy ` | 81348 |
| 65 | ` Whitney ` | 66097 |
| 66 | ` Suffolk ` | 80783 |
| 67 | ` 新西兰 ` | 109751 |
| 68 | ` Tyler ` | 31220 |
| 69 | ` Amanda ` | 41759 |
| 70 | ` 让她 ` | 105655 |
| 71 | ` Lauren ` | 42360 |
| 72 | ` 蓑 ` | 121132 |
| 73 | ` Christina ` | 59671 |
| 74 | ` Oregon ` | 18784 |
| 75 | ` Cindy ` | 69331 |
| 76 | ` 汕头 ` | 112936 |
| 77 | ` Dorothy ` | 72558 |
| 78 | ` 璎 ` | 121546 |
| 79 | ` 广州市 ` | 106805 |
| 80 | ` Cameron ` | 26426 |
| 81 | ` 告诉她 ` | 117154 |
| 82 | ` ("$. ` | 91699 |
| 83 | ` 清远 ` | 117316 |
| 84 | ` Olivia ` | 59287 |
| 85 | ` ** ` | 334 |
| 86 | ` Elliot ` | 68418 |
| 87 | ` Cait ` | 75332 |
| 88 | ` "                     " ` | 2549 |
| 89 | ` "                       " ` | 664 |
| 90 | ` "                          " ` | 5968 |
| 91 | ` Edward ` | 84501 |
| 92 | ` 捐赠 ` | 106120 |
| 93 | ` "                                                      " ` | 25773 |
| 94 | ` 莉 ` | 105050 |
| 95 | ` 浈 ` | 122522 |
| 96 | ` Quinn ` | 40523 |
| 97 | ` Esther ` | 83638 |
| 98 | ` **/ ` | 88490 |
| 99 | ` :** ` | 66963 |
| 100 | ` Patricia ` | 53934 |

</details>

### Row 11: Michael

<details>
<summary>Michael: predictions 1–100 · target man · target rank 30,483 (0-based)</summary>

| Prediction rank | J-lens token | Token ID |
| ---: | --- | ---: |
| 1 | ` Austrian ` | 56969 |
| 2 | ` Michael ` | 7937 |
| 3 | ` Wolfgang ` | 86498 |
| 4 | ` Jewish ` | 13350 |
| 5 | ` beach ` | 11321 |
| 6 | ` Judaism ` | 63183 |
| 7 | ` marine ` | 28591 |
| 8 | ` Swiss ` | 29691 |
| 9 | ` German ` | 5938 |
| 10 | ` Paul ` | 6898 |
| 11 | ` Jews ` | 16987 |
| 12 | ` Philippe ` | 66854 |
| 13 | ` Michael ` | 25516 |
| 14 | ` Austria ` | 34898 |
| 15 | ` Germany ` | 9856 |
| 16 | ` seaside ` | 92776 |
| 17 | ` French ` | 8585 |
| 18 | ` Saul ` | 67524 |
| 19 | ` Stephen ` | 18095 |
| 20 | ` jewish ` | 88138 |
| 21 | ` Germans ` | 40937 |
| 22 | ` Hawaiian ` | 58003 |
| 23 | ` σ ` | 43928 |
| 24 | ` Holocaust ` | 43098 |
| 25 | ` σ ` | 47723 |
| 26 | ` Catholics ` | 50871 |
| 27 | ` german ` | 42527 |
| 28 | ` thyroid ` | 52960 |
| 29 | ` Friedrich ` | 79374 |
| 30 | ` sea ` | 9396 |
| 31 | ` shore ` | 28426 |
| 32 | ` Ludwig ` | 84450 |
| 33 | ` Hitler ` | 30554 |
| 34 | ` Deutsch ` | 33315 |
| 35 | ` Polish ` | 31984 |
| 36 | ` Albert ` | 17513 |
| 37 | ` seaw ` | 66229 |
| 38 | ` kosher ` | 97640 |
| 39 | ` Marine ` | 22963 |
| 40 | ` Baltic ` | 71989 |
| 41 | ` Switzerland ` | 29121 |
| 42 | ` surfing ` | 57740 |
| 43 | ` french ` | 41193 |
| 44 | ` Paulo ` | 36926 |
| 45 | ` lobster ` | 79715 |
| 46 | ` michael ` | 87906 |
| 47 | ` Auschwitz ` | 92290 |
| 48 | ` … ` | 89478 |
| 49 | ` Paul ` | 25300 |
| 50 | ` ocean ` | 17951 |
| 51 | ` Wilhelm ` | 92437 |
| 52 | ` λ ` | 48338 |
| 53 | ` shoreline ` | 98064 |
| 54 | ` Catholic ` | 16473 |
| 55 | ` Peter ` | 11044 |
| 56 | ` German ` | 32079 |
| 57 | ` λ ` | 33486 |
| 58 | ` Adolf ` | 89485 |
| 59 | ` NYPD ` | 73155 |
| 60 | ` Georges ` | 94259 |
| 61 | ` beaches ` | 34809 |
| 62 | ` France ` | 9625 |
| 63 | ` Andrew ` | 13615 |
| 64 | ` compuls ` | 47106 |
| 65 | ` shore ` | 30184 |
| 66 | ` André ` | 84800 |
| 67 | ` Pennsylvania ` | 19771 |
| 68 | ` marine ` | 84443 |
| 69 | ` HPV ` | 73369 |
| 70 | ` Klein ` | 42141 |
| 71 | ` Becker ` | 77384 |
| 72 | ` ドイツ ` | 142765 |
| 73 | ` Germany ` | 50170 |
| 74 | ` Philip ` | 25179 |
| 75 | ` SCH ` | 74838 |
| 76 | ` Lithuania ` | 70506 |
| 77 | ` Hawaii ` | 27521 |
| 78 | ` Jerry ` | 28708 |
| 79 | ` Albert ` | 66622 |
| 80 | ` Marie ` | 32016 |
| 81 | ` Jew ` | 7958 |
| 82 | ` aph ` | 88011 |
| 83 | ` synthes ` | 51289 |
| 84 | ` seab ` | 65491 |
| 85 | ` Vienna ` | 46287 |
| 86 | ` reef ` | 70045 |
| 87 | ` Christoph ` | 94593 |
| 88 | ` clar ` | 19497 |
| 89 | ` HK ` | 42217 |
| 90 | ` Belgian ` | 48062 |
| 91 | ` paul ` | 67011 |
| 92 | ` Franco ` | 43843 |
| 93 | ` Philip ` | 78644 |
| 94 | ` catholic ` | 82091 |
| 95 | ` Maryland ` | 22652 |
| 96 | ` coastal ` | 34235 |
| 97 | ` seafood ` | 52305 |
| 98 | ` Hungarian ` | 56769 |
| 99 | ` swim ` | 16191 |
| 100 | ` Andreas ` | 50062 |

</details>

## Provenance

The original measurement is retained in [the unchanged snapshot](name_gender_measurement.json). The new JSON records checkpoint and source hashes, library versions, device, and numerical verification. [The extraction script](export_name_gender_top100.py) uses the original measurement workspace’s `jlens.model_sweep` implementation and checks its source hashes before running; those measurement modules and the cached model checkpoint are required to rerun inference.
