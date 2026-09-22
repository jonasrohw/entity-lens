# Qwen3 8B Base: name_gender with the Facts about prompt

**Exact prompt: `Facts about {name}:`** — capital F, colon after the name, and no leading newline. These are 100 differential J-lens predictions for each of the same 11 saved examples (1,100 entries). Sofia is retained twice.

[CSV: all predictions](name_gender_facts_about_top100.csv) · [JSON: scores and provenance](name_gender_facts_about_top100.json) · [Compare the newline-prompt tables](name_gender_top100.md)

## Top-five overview and target ranks

| Row | Name / top 100 | Model target | J-lens top five | Facts about target rank | Previous newline target rank |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | [Tom](#row-1-tom) | ` man ` | ` 西红柿 `, ` 番茄 `, ` tomato `, ` 萝卜 `, ` 鄱 ` | 96,047 | 60,513 |
| 2 | [Mia](#row-2-mia) | ` woman ` | ` Mia `, ` actress `, ` 她 `, ` 她是 `, ` 她在 ` | 458 | 108,458 |
| 3 | [Sofia](#row-3-sofia) | ` woman ` | ` Sofia `, ` Bulgaria `, ` Bulgarian `, ` 基辅 `, ` Estonia ` | 5,059 | 9,365 |
| 4 | [Connor](#row-4-connor) | ` man ` | ` Conor `, ` Connor `, ` 年龄 `, ` age `, ` Liam ` | 19,142 | 101,074 |
| 5 | [Evan](#row-5-evan) | ` man ` | ` Evan `, ` Winnipeg `, ` Devin `, ` Seth `, ` Chicago ` | 34,109 | 43,448 |
| 6 | [Sofia](#row-6-sofia) | ` woman ` | ` Sofia `, ` Bulgaria `, ` Bulgarian `, ` 基辅 `, ` Estonia ` | 5,059 | 9,365 |
| 7 | [Natalie](#row-7-natalie) | ` woman ` | ` actress `, ` Natalie `, ` Actress `, ` singer `, ` natal ` | 61 | 22,301 |
| 8 | [David](#row-8-david) | ` man ` | ` Saul `, ` prophet `, ` 泰州 `, ` novelist `, ` theolog ` | 86,813 | 11,338 |
| 9 | [Dylan](#row-9-dylan) | ` man ` | ` Dylan `, ` songwriter `, ` 歌手 `, ` singer `, ` 诗人 ` | 83,659 | 110,309 |
| 10 | [Emily](#row-10-emily) | ` woman ` | ` Emily `, ` Emily `, ` 作家 `, ` poet `, ` Dickinson ` | 74 | 98,417 |
| 11 | [Michael](#row-11-michael) | ` man ` | ` Psychiat `, ` actress `, ` psychiatrist `, ` ochond `, ` actor ` | 74,755 | 30,483 |

**Target ranks above are zero-based** (0 = top-1). The ranked lists below are **one-based** (1 = the highest-scoring prediction). The target remains the exact full model top-1 token under the relation prompt `{name} is usually a name for a` with the original in-context examples.

## Summary

| Quantity | Facts about | Previous newline |
| --- | ---: | ---: |
| Exact target in top 1 | 0 / 11 | 0 / 11 |
| Exact target in top 10 | 0 / 11 | 0 / 11 |
| Exact target in top 100 | 2 / 11 | 0 / 11 |
| Median target rank (zero-based) | 19,142.000000 | 43,448.000000 |
| Representation score | -4.282010 | -4.637980 |

A male-related Chinese token appears in this prompt variant: **Connor: 男孩 (boy), prediction #88** (one-based).

The model, checkpoint, state 19, dataset rows, targets, and reference counts are unchanged. This rerun uses the earlier Facts about run’s float32/eager attention and batch size 2. The previous newline export used batch size 8. **The complete rerun exactly reproduces the earlier saved Facts about measurement**, including its per-row ranks, top-five tokens, and aggregates. LRE faithfulness remains 1.0 on its separate three test rows; it was not rerun. The existing 26-relation plot continues to show the newline results.

## Top-100 tables

Expand a name for all 100 predictions. Token strings are shown with surrounding whitespace stripped; IDs distinguish tokens with the same visible text. The CSV `token_json` field preserves the exact decoded string with JSON escapes; the JSON also contains unstripped strings and raw readout scores, which are not probabilities. A check mark identifies an exact match to the model target token ID.

### Row 1: Tom

<details open>
<summary>Tom: predictions 1–100 · target man · target rank 96,047 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` 西红柿 ` | 117700 |  |
| 2 | ` 番茄 ` | 113400 |  |
| 3 | ` tomato ` | 41020 |  |
| 4 | ` 萝卜 ` | 103438 |  |
| 5 | ` 鄱 ` | 121470 |  |
| 6 | ` 红薯 ` | 117146 |  |
| 7 | ` Tomato ` | 92991 |  |
| 8 | ` 作家 ` | 105022 |  |
| 9 | ` 小说 ` | 104032 |  |
| 10 | ` 渔民 ` | 116613 |  |
| 11 | ` tomatoes ` | 40513 |  |
| 12 | ` 晚年 ` | 114836 |  |
| 13 | ` 鲁迅 ` | 113183 |  |
| 14 | ` novelist ` | 80647 |  |
| 15 | ` 白菜 ` | 111822 |  |
| 16 | ` 赊 ` | 120688 |  |
| 17 | ` 养猪 ` | 116231 |  |
| 18 | ` 鸵 ` | 119387 |  |
| 19 | ` 墨西哥 ` | 109448 |  |
| 20 | ` TOM ` | 82465 |  |
| 21 | ` 狸 ` | 103702 |  |
| 22 | ` 🍅 ` | 146824 |  |
| 23 | ` Pom ` | 38312 |  |
| 24 | ` 茆 ` | 119826 |  |
| 25 | ` gtk ` | 43051 |  |
| 26 | ` 苕 ` | 119832 |  |
| 27 | ` Tomas ` | 87936 |  |
| 28 | ` 圩 ` | 119033 |  |
| 29 | ` 婶 ` | 118504 |  |
| 30 | ` 蒿 ` | 119245 |  |
| 31 | ` 豇 ` | 120341 |  |
| 32 | ` 江西省 ` | 109040 |  |
| 33 | ` đò ` | 138584 |  |
| 34 | ` 媳妇 ` | 108810 |  |
| 35 | ` 产品经理 ` | 118223 |  |
| 36 | ` 西湖 ` | 110192 |  |
| 37 | ` 农 ` | 99288 |  |
| 38 | ` 柿 ` | 103847 |  |
| 39 | ` 篙 ` | 121789 |  |
| 40 | ` /docker ` | 61764 |  |
| 41 | ` 作者 ` | 57421 |  |
| 42 | ` 西红 ` | 117699 |  |
| 43 | ` 煲 ` | 114728 |  |
| 44 | ` 匡 ` | 117356 |  |
| 45 | ` pom ` | 29484 |  |
| 46 | ` 生日 ` | 105511 |  |
| 47 | ` 和他的 ` | 112370 |  |
| 48 | ` 徐州 ` | 108457 |  |
| 49 | ` 蓉 ` | 103187 |  |
| 50 | ` 因为他 ` | 106773 |  |
| 51 | ` 理发 ` | 117516 |  |
| 52 | ` 浏 ` | 101525 |  |
| 53 | ` 柁 ` | 120070 |  |
| 54 | ` 江西 ` | 104732 |  |
| 55 | ` 苹 ` | 102081 |  |
| 56 | ` 身高 ` | 107958 |  |
| 57 | ` 涿 ` | 120777 |  |
| 58 | ` .docker ` | 91131 |  |
| 59 | ` docker ` | 28648 |  |
| 60 | ` Mexicans ` | 96762 |  |
| 61 | ` 酱 ` | 102675 |  |
| 62 | ` 的妻子 ` | 111252 |  |
| 63 | ` 出身 ` | 107912 |  |
| 64 | ` 出生于 ` | 112625 |  |
| 65 | ` 胡萝卜 ` | 111761 |  |
| 66 | ` 程序员 ` | 118552 |  |
| 67 | ` 猢 ` | 121007 |  |
| 68 | ` 景德镇 ` | 117421 |  |
| 69 | ` 抽烟 ` | 115980 |  |
| 70 | ` 妻子 ` | 102375 |  |
| 71 | ` 退休 ` | 102678 |  |
| 72 | ` 彘 ` | 121087 |  |
| 73 | ` 娶 ` | 107519 |  |
| 74 | ` 红楼梦 ` | 117805 |  |
| 75 | ` 柞 ` | 120064 |  |
| 76 | ` Dickens ` | 92899 |  |
| 77 | ` 民国 ` | 108982 |  |
| 78 | ` 猪 ` | 100761 |  |
| 79 | ` 孙悟空 ` | 116250 |  |
| 80 | ` 渔 ` | 100938 |  |
| 81 | ` 泰州 ` | 117053 |  |
| 82 | ` 梃 ` | 120328 |  |
| 83 | ` 红茶 ` | 117653 |  |
| 84 | ` 女朋友 ` | 108508 |  |
| 85 | ` 厨师 ` | 114396 |  |
| 86 | ` 橡 ` | 103218 |  |
| 87 | ` 舵 ` | 119029 |  |
| 88 | ` 帮他 ` | 115072 |  |
| 89 | ` 豆腐 ` | 107590 |  |
| 90 | ` writer ` | 38959 |  |
| 91 | ` Tommy ` | 39039 |  |
| 92 | ` actor ` | 93410 |  |
| 93 | ` 祖父 ` | 117378 |  |
| 94 | ` farmer ` | 36400 |  |
| 95 | ` 农家 ` | 103613 |  |
| 96 | ` 小說 ` | 115463 |  |
| 97 | ` 嫂 ` | 109790 |  |
| 98 | ` 盐城 ` | 115830 |  |
| 99 | ` 宿迁 ` | 117446 |  |
| 100 | ` 剃 ` | 118895 |  |

</details>

### Row 2: Mia

<details>
<summary>Mia: predictions 1–100 · target woman · target rank 458 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Mia ` | 60597 |  |
| 2 | ` actress ` | 23654 |  |
| 3 | ` 她 ` | 99517 |  |
| 4 | ` 她是 ` | 107852 |  |
| 5 | ` 她在 ` | 108333 |  |
| 6 | ` Actress ` | 78439 |  |
| 7 | ` she ` | 1340 |  |
| 8 | ` 演员 ` | 104534 |  |
| 9 | ` Icelandic ` | 99148 |  |
| 10 | ` 女主角 ` | 114335 |  |
| 11 | ` Miami ` | 17584 |  |
| 12 | ` She ` | 2932 |  |
| 13 | ` actresses ` | 90028 |  |
| 14 | ` 她的 ` | 104007 |  |
| 15 | ` Swedish ` | 30109 |  |
| 16 | ` 但她 ` | 110204 |  |
| 17 | ` 导演 ` | 104727 |  |
| 18 | ` 女孩 ` | 101339 |  |
| 19 | ` 年龄 ` | 102185 |  |
| 20 | ` 女 ` | 57750 |  |
| 21 | ` 模特 ` | 110016 |  |
| 22 | ` 瑞典 ` | 110055 |  |
| 23 | ` 女主 ` | 111938 |  |
| 24 | ` Gio ` | 66665 |  |
| 25 | ` 三亚 ` | 107442 |  |
| 26 | ` 对她 ` | 108663 |  |
| 27 | ` 因为她 ` | 113057 |  |
| 28 | ` 给她 ` | 106967 |  |
| 29 | ` ella ` | 53501 |  |
| 30 | ` 女子 ` | 104340 |  |
| 31 | ` nàng ` | 130453 |  |
| 32 | ` 信阳 ` | 117643 |  |
| 33 | ` 天府 ` | 113291 |  |
| 34 | ` 成都 ` | 104051 |  |
| 35 | ` 海口 ` | 107034 |  |
| 36 | ` Napoli ` | 85676 |  |
| 37 | ` Leia ` | 87329 |  |
| 38 | ` elle ` | 26451 |  |
| 39 | ` She ` | 7941 |  |
| 40 | ` 她说 ` | 106095 |  |
| 41 | ` 主演 ` | 107119 |  |
| 42 | ` 的女孩 ` | 111374 |  |
| 43 | ` 小女孩 ` | 112060 |  |
| 44 | ` Shea ` | 84968 |  |
| 45 | ` Naples ` | 82621 |  |
| 46 | ` mia ` | 56253 |  |
| 47 | ` 郑州 ` | 104839 |  |
| 48 | ` Female ` | 28638 |  |
| 49 | ` 儋 ` | 121643 |  |
| 50 | ` 编剧 ` | 111346 |  |
| 51 | ` she ` | 31058 |  |
| 52 | ` 成都市 ` | 109581 |  |
| 53 | ` 性别 ` | 109391 |  |
| 54 | ` Lua ` | 37662 |  |
| 55 | ` Hawaiian ` | 58003 |  |
| 56 | ` Artist ` | 28359 |  |
| 57 | ` herself ` | 11675 |  |
| 58 | ` 女性 ` | 101968 |  |
| 59 | ` girl ` | 3743 |  |
| 60 | ` female ` | 8778 |  |
| 61 | ` 让她 ` | 105655 |  |
| 62 | ` 女生 ` | 104535 |  |
| 63 | ` Mädchen ` | 78596 |  |
| 64 | ` 身高 ` | 107958 |  |
| 65 | ` 郑州市 ` | 113149 |  |
| 66 | ` 把她 ` | 110284 |  |
| 67 | ` blonde ` | 26026 |  |
| 68 | ` 芈 ` | 122215 |  |
| 69 | ` Director ` | 10560 |  |
| 70 | ` 出演 ` | 111162 |  |
| 71 | ` Shia ` | 77409 |  |
| 72 | ` Latina ` | 58763 |  |
| 73 | ` 饰演 ` | 108316 |  |
| 74 | ` princess ` | 39588 |  |
| 75 | ` 闺蜜 ` | 113006 |  |
| 76 | ` .She ` | 98541 |  |
| 77 | ` Sao ` | 89872 |  |
| 78 | ` 出生于 ` | 112625 |  |
| 79 | ` Femme ` | 57525 |  |
| 80 | ` 绵阳 ` | 114348 |  |
| 81 | ` 剧组 ` | 116360 |  |
| 82 | ` actor ` | 12089 |  |
| 83 | ` singer ` | 22757 |  |
| 84 | ` 妞 ` | 114049 |  |
| 85 | ` .lua ` | 37011 |  |
| 86 | ` 创作者 ` | 118297 |  |
| 87 | ` 公主 ` | 105454 |  |
| 88 | ` Actors ` | 81361 |  |
| 89 | ` director ` | 7538 |  |
| 90 | ` Bian ` | 66945 |  |
| 91 | ` 女星 ` | 114224 |  |
| 92 | ` Chloe ` | 59370 |  |
| 93 | ` her ` | 1059 |  |
| 94 | ` Sweden ` | 23190 |  |
| 95 | ` pio ` | 11917 |  |
| 96 | ` Maui ` | 92450 |  |
| 97 | ` Svens ` | 86173 |  |
| 98 | ` Composer ` | 78771 |  |
| 99 | ` 女孩子 ` | 107938 |  |
| 100 | ` 杨幂 ` | 115680 |  |

</details>

### Row 3: Sofia

<details>
<summary>Sofia: predictions 1–100 · target woman · target rank 5,059 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Sofia ` | 78563 |  |
| 2 | ` Bulgaria ` | 58719 |  |
| 3 | ` Bulgarian ` | 88624 |  |
| 4 | ` 基辅 ` | 117572 |  |
| 5 | ` Estonia ` | 68362 |  |
| 6 | ` Romania ` | 46049 |  |
| 7 | ` Ukrainian ` | 33625 |  |
| 8 | ` Albania ` | 94633 |  |
| 9 | ` Romanian ` | 73597 |  |
| 10 | ` Ukraine ` | 18753 |  |
| 11 | ` Kiev ` | 61625 |  |
| 12 | ` Czech ` | 33150 |  |
| 13 | ` Prague ` | 67289 |  |
| 14 | ` Serbia ` | 58714 |  |
| 15 | ` Lithuania ` | 70506 |  |
| 16 | ` Croatia ` | 57211 |  |
| 17 | ` Kosovo ` | 79897 |  |
| 18 | ` Ukrain ` | 30604 |  |
| 19 | ` Macedonia ` | 76409 |  |
| 20 | ` Serbian ` | 87164 |  |
| 21 | ` Eston ` | 53323 |  |
| 22 | ` Russia ` | 44506 |  |
| 23 | ` Budapest ` | 69595 |  |
| 24 | ` Slovakia ` | 80685 |  |
| 25 | ` Portugal ` | 33311 |  |
| 26 | ` 乌克兰 ` | 105113 |  |
| 27 | ` Latvia ` | 78158 |  |
| 28 | ` Azerbai ` | 64323 |  |
| 29 | ` Bulg ` | 41250 |  |
| 30 | ` Croatian ` | 99070 |  |
| 31 | ` Slovenia ` | 78217 |  |
| 32 | ` Russia ` | 8359 |  |
| 33 | ` Uzbek ` | 89859 |  |
| 34 | ` Sophia ` | 70371 |  |
| 35 | ` Crimea ` | 60711 |  |
| 36 | ` Sweden ` | 96871 |  |
| 37 | ` Catalonia ` | 95852 |  |
| 38 | ` Portuguese ` | 42188 |  |
| 39 | ` Lisbon ` | 80701 |  |
| 40 | ` Hungary ` | 42776 |  |
| 41 | ` Azerbaijan ` | 73477 |  |
| 42 | ` .cz ` | 74504 |  |
| 43 | ` Wrocław ` | 139222 |  |
| 44 | ` Ukr ` | 14299 |  |
| 45 | ` Sevilla ` | 94894 |  |
| 46 | ` Georgian ` | 86442 |  |
| 47 | ` Colombia ` | 38033 |  |
| 48 | ` Poland ` | 27602 |  |
| 49 | ` Укра ` | 127872 |  |
| 50 | ` Copenhagen ` | 63061 |  |
| 51 | ` Bosnia ` | 76328 |  |
| 52 | ` Mongolia ` | 90750 |  |
| 53 | ` Donetsk ` | 97351 |  |
| 54 | ` İstanbul ` | 129514 |  |
| 55 | ` Russian ` | 47707 |  |
| 56 | ` Armenia ` | 71530 |  |
| 57 | ` Belarus ` | 69506 |  |
| 58 | ` Slovak ` | 61264 |  |
| 59 | ` украин ` | 138132 |  |
| 60 | ` Spain ` | 17689 |  |
| 61 | ` Finland ` | 36255 |  |
| 62 | ` Sweden ` | 23190 |  |
| 63 | ` Russians ` | 33966 |  |
| 64 | ` Madrid ` | 24081 |  |
| 65 | ` Peru ` | 47747 |  |
| 66 | ` Russian ` | 8522 |  |
| 67 | ` Honduras ` | 70315 |  |
| 68 | ` Moscow ` | 22415 |  |
| 69 | ` Swedish ` | 30109 |  |
| 70 | ` russian ` | 63145 |  |
| 71 | ` NATO ` | 29234 |  |
| 72 | ` Киев ` | 138868 |  |
| 73 | ` Denmark ` | 34340 |  |
| 74 | ` Helsinki ` | 76334 |  |
| 75 | ` Hungarian ` | 56769 |  |
| 76 | ` Yugosl ` | 74451 |  |
| 77 | ` 莫斯科 ` | 112969 |  |
| 78 | ` Gdańsk ` | 141845 |  |
| 79 | ` Alban ` | 56891 |  |
| 80 | ` Nordic ` | 63366 |  |
| 81 | ` Venezuelan ` | 82649 |  |
| 82 | ` Maced ` | 56452 |  |
| 83 | ` Bolivia ` | 76610 |  |
| 84 | ` Elena ` | 68575 |  |
| 85 | ` Italy ` | 15344 |  |
| 86 | ` sof ` | 85806 |  |
| 87 | ` Istanbul ` | 53368 |  |
| 88 | ` -Russian ` | 84035 |  |
| 89 | ` Spain ` | 74320 |  |
| 90 | ` Venezuela ` | 35903 |  |
| 91 | ` Petersburg ` | 53948 |  |
| 92 | ` Украин ` | 137829 |  |
| 93 | ` cityName ` | 96701 |  |
| 94 | ` 公主 ` | 105454 |  |
| 95 | ` Colombian ` | 83553 |  |
| 96 | ` 葡萄牙 ` | 112334 |  |
| 97 | ` Sof ` | 35547 |  |
| 98 | ` .ua ` | 77695 |  |
| 99 | ` Россия ` | 137421 |  |
| 100 | ` Kremlin ` | 50324 |  |

</details>

### Row 4: Connor

<details>
<summary>Connor: predictions 1–100 · target man · target rank 19,142 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Conor ` | 84115 |  |
| 2 | ` Connor ` | 57180 |  |
| 3 | ` 年龄 ` | 102185 |  |
| 4 | ` age ` | 4231 |  |
| 5 | ` Liam ` | 60642 |  |
| 6 | ` Age ` | 13081 |  |
| 7 | ` Connor ` | 56887 |  |
| 8 | ` Belfast ` | 54239 |  |
| 9 | ` Newfoundland ` | 88296 |  |
| 10 | ` Canucks ` | 80502 |  |
| 11 | ` Nolan ` | 61632 |  |
| 12 | ` Joey ` | 54735 |  |
| 13 | ` siblings ` | 36683 |  |
| 14 | ` Age ` | 16749 |  |
| 15 | ` Winnipeg ` | 51882 |  |
| 16 | ` Devin ` | 79992 |  |
| 17 | ` 年龄段 ` | 117372 |  |
| 18 | ` Bannon ` | 54165 |  |
| 19 | ` gender ` | 9825 |  |
| 20 | ` Caleb ` | 83062 |  |
| 21 | ` ages ` | 16639 |  |
| 22 | ` Finn ` | 34062 |  |
| 23 | ` Quinn ` | 40523 |  |
| 24 | ` age ` | 424 |  |
| 25 | ` Fortnite ` | 86420 |  |
| 26 | ` 男孩子 ` | 118842 |  |
| 27 | ` Toronto ` | 14632 |  |
| 28 | ` 性别 ` | 109391 |  |
| 29 | ` _age ` | 32053 |  |
| 30 | ` Twins ` | 65268 |  |
| 31 | ` Edmonton ` | 45287 |  |
| 32 | ` hockey ` | 26953 |  |
| 33 | ` Swagger ` | 79631 |  |
| 34 | ` Gender ` | 28217 |  |
| 35 | ` aged ` | 19749 |  |
| 36 | ` Icelandic ` | 99148 |  |
| 37 | ` Brandon ` | 31766 |  |
| 38 | ` Parents ` | 42926 |  |
| 39 | ` Loki ` | 83755 |  |
| 40 | ` Male ` | 19396 |  |
| 41 | ` Brendan ` | 55843 |  |
| 42 | ` toddler ` | 51235 |  |
| 43 | ` Fiona ` | 88843 |  |
| 44 | ` goalie ` | 73333 |  |
| 45 | ` Cassidy ` | 91797 |  |
| 46 | ` Sex ` | 6695 |  |
| 47 | ` Liverpool ` | 24688 |  |
| 48 | ` Devon ` | 59334 |  |
| 49 | ` Fallon ` | 87068 |  |
| 50 | ` Tanner ` | 89796 |  |
| 51 | ` Dublin ` | 32877 |  |
| 52 | ` blonde ` | 26026 |  |
| 53 | ` goalt ` | 79622 |  |
| 54 | ` Ages ` | 48993 |  |
| 55 | ` Kodi ` | 81312 |  |
| 56 | ` Irish ` | 17623 |  |
| 57 | ` Vancouver ` | 22575 |  |
| 58 | ` male ` | 8593 |  |
| 59 | ` Clan ` | 54595 |  |
| 60 | ` Cait ` | 75332 |  |
| 61 | ` Owen ` | 46709 |  |
| 62 | ` Brennan ` | 58750 |  |
| 63 | ` Player ` | 7312 |  |
| 64 | ` sex ` | 1839 |  |
| 65 | ` Donovan ` | 70700 |  |
| 66 | ` Girlfriend ` | 98493 |  |
| 67 | ` Welsh ` | 45781 |  |
| 68 | ` younger ` | 14650 |  |
| 69 | ` masculinity ` | 79711 |  |
| 70 | ` 帅气 ` | 113176 |  |
| 71 | ` parents ` | 6562 |  |
| 72 | ` -age ` | 42941 |  |
| 73 | ` Sheffield ` | 60025 |  |
| 74 | ` born ` | 9223 |  |
| 75 | ` 患儿 ` | 116582 |  |
| 76 | ` baby ` | 8770 |  |
| 77 | ` pokemon ` | 57380 |  |
| 78 | ` RCMP ` | 74624 |  |
| 79 | ` Soccer ` | 37533 |  |
| 80 | ` Hanson ` | 75213 |  |
| 81 | ` Corey ` | 54652 |  |
| 82 | ` males ` | 24037 |  |
| 83 | ` adorable ` | 40608 |  |
| 84 | ` Gavin ` | 63395 |  |
| 85 | ` ages ` | 1134 |  |
| 86 | ` Hockey ` | 40242 |  |
| 87 | ` Boys ` | 29757 |  |
| 88 | ` 男孩 ` | 102760 |  |
| 89 | ` boys ` | 13007 |  |
| 90 | ` 隰 ` | 121842 |  |
| 91 | ` Ontario ` | 20551 |  |
| 92 | ` pediatric ` | 52066 |  |
| 93 | ` Overwatch ` | 72338 |  |
| 94 | ` His ` | 5301 |  |
| 95 | ` Injury ` | 65076 |  |
| 96 | ` AGE ` | 90588 |  |
| 97 | ` Labrador ` | 79276 |  |
| 98 | ` Gender ` | 28914 |  |
| 99 | ` 垚 ` | 122400 |  |
| 100 | ` Dylan ` | 43358 |  |

</details>

### Row 5: Evan

<details>
<summary>Evan: predictions 1–100 · target man · target rank 34,109 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Evan ` | 43943 |  |
| 2 | ` Winnipeg ` | 51882 |  |
| 3 | ` Devin ` | 79992 |  |
| 4 | ` Seth ` | 44218 |  |
| 5 | ` Chicago ` | 10557 |  |
| 6 | ` Vancouver ` | 22575 |  |
| 7 | ` SVC ` | 88212 |  |
| 8 | ` Devon ` | 59334 |  |
| 9 | ` Denver ` | 22117 |  |
| 10 | ` Chicago ` | 46175 |  |
| 11 | ` Vermont ` | 34639 |  |
| 12 | ` Vegan ` | 62115 |  |
| 13 | ` Irvine ` | 71285 |  |
| 14 | ` Colorado ` | 15383 |  |
| 15 | ` Brooke ` | 78461 |  |
| 16 | ` Haley ` | 67711 |  |
| 17 | ` Cleveland ` | 23475 |  |
| 18 | ` Illinois ` | 18657 |  |
| 19 | ` chicago ` | 92208 |  |
| 20 | ` Ethan ` | 62164 |  |
| 21 | ` 芝加哥 ` | 118390 |  |
| 22 | ` Corey ` | 54652 |  |
| 23 | ` Donovan ` | 70700 |  |
| 24 | ` Brighton ` | 57419 |  |
| 25 | ` Halo ` | 45249 |  |
| 26 | ` Blackhawks ` | 92392 |  |
| 27 | ` 汕 ` | 103600 |  |
| 28 | ` Charleston ` | 52293 |  |
| 29 | ` Raven ` | 45107 |  |
| 30 | ` Newport ` | 58800 |  |
| 31 | ` Brandon ` | 31766 |  |
| 32 | ` vegan ` | 29149 |  |
| 33 | ` Canucks ` | 80502 |  |
| 34 | ` Seattle ` | 16355 |  |
| 35 | ` Halifax ` | 69669 |  |
| 36 | ` Tucson ` | 66852 |  |
| 37 | ` singer ` | 22757 |  |
| 38 | ` Minnesota ` | 18925 |  |
| 39 | ` Manitoba ` | 63240 |  |
| 40 | ` Ontario ` | 20551 |  |
| 41 | ` Utah ` | 22389 |  |
| 42 | ` Manit ` | 53235 |  |
| 43 | ` 肇庆 ` | 118070 |  |
| 44 | ` Tanner ` | 89796 |  |
| 45 | ` Iowa ` | 20700 |  |
| 46 | ` Canyon ` | 40460 |  |
| 47 | ` Fargo ` | 57650 |  |
| 48 | ` 年龄 ` | 102185 |  |
| 49 | ` Scot ` | 15468 |  |
| 50 | ` Indianapolis ` | 41351 |  |
| 51 | ` Cameron ` | 26426 |  |
| 52 | ` age ` | 4231 |  |
| 53 | ` songwriter ` | 84657 |  |
| 54 | ` Hawaiian ` | 58003 |  |
| 55 | ` lake ` | 62110 |  |
| 56 | ` Marin ` | 63377 |  |
| 57 | ` Chloe ` | 59370 |  |
| 58 | ` Minnesota ` | 92365 |  |
| 59 | ` sophomore ` | 50398 |  |
| 60 | ` 疍 ` | 122724 |  |
| 61 | ` Brendan ` | 55843 |  |
| 62 | ` 谌 ` | 120798 |  |
| 63 | ` Benton ` | 92683 |  |
| 64 | ` Leeds ` | 51529 |  |
| 65 | ` Shawn ` | 55537 |  |
| 66 | ` Wisconsin ` | 20441 |  |
| 67 | ` svc ` | 46154 |  |
| 68 | ` Shea ` | 84968 |  |
| 69 | ` Chad ` | 42030 |  |
| 70 | ` Vanessa ` | 70519 |  |
| 71 | ` Colorado ` | 77469 |  |
| 72 | ` 郴州 ` | 118608 |  |
| 73 | ` Vander ` | 58042 |  |
| 74 | ` Nevada ` | 26868 |  |
| 75 | ` Vance ` | 91268 |  |
| 76 | ` Palo ` | 84151 |  |
| 77 | ` Scott ` | 9815 |  |
| 78 | ` 岚 ` | 109512 |  |
| 79 | ` Hampton ` | 63674 |  |
| 80 | ` Indiana ` | 21591 |  |
| 81 | ` MVP ` | 31866 |  |
| 82 | ` Rochester ` | 52368 |  |
| 83 | ` Gavin ` | 63395 |  |
| 84 | ` 毕业于 ` | 109980 |  |
| 85 | ` vocalist ` | 95565 |  |
| 86 | ` rvine ` | 66574 |  |
| 87 | ` SVM ` | 90009 |  |
| 88 | ` Maui ` | 92450 |  |
| 89 | ` Tottenham ` | 55548 |  |
| 90 | ` Seattle ` | 71672 |  |
| 91 | ` Heather ` | 46263 |  |
| 92 | ` Indie ` | 76339 |  |
| 93 | ` Caleb ` | 83062 |  |
| 94 | ` Quinn ` | 40523 |  |
| 95 | ` Ukrainian ` | 33625 |  |
| 96 | ` nv ` | 31440 |  |
| 97 | ` Denver ` | 95201 |  |
| 98 | ` sean ` | 84422 |  |
| 99 | ` Purdue ` | 72234 |  |
| 100 | ` Appalachian ` | 97927 |  |

</details>

### Row 6: Sofia

<details>
<summary>Sofia: predictions 1–100 · target woman · target rank 5,059 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Sofia ` | 78563 |  |
| 2 | ` Bulgaria ` | 58719 |  |
| 3 | ` Bulgarian ` | 88624 |  |
| 4 | ` 基辅 ` | 117572 |  |
| 5 | ` Estonia ` | 68362 |  |
| 6 | ` Romania ` | 46049 |  |
| 7 | ` Ukrainian ` | 33625 |  |
| 8 | ` Albania ` | 94633 |  |
| 9 | ` Romanian ` | 73597 |  |
| 10 | ` Ukraine ` | 18753 |  |
| 11 | ` Kiev ` | 61625 |  |
| 12 | ` Czech ` | 33150 |  |
| 13 | ` Prague ` | 67289 |  |
| 14 | ` Serbia ` | 58714 |  |
| 15 | ` Lithuania ` | 70506 |  |
| 16 | ` Croatia ` | 57211 |  |
| 17 | ` Kosovo ` | 79897 |  |
| 18 | ` Ukrain ` | 30604 |  |
| 19 | ` Macedonia ` | 76409 |  |
| 20 | ` Serbian ` | 87164 |  |
| 21 | ` Eston ` | 53323 |  |
| 22 | ` Russia ` | 44506 |  |
| 23 | ` Budapest ` | 69595 |  |
| 24 | ` Slovakia ` | 80685 |  |
| 25 | ` Portugal ` | 33311 |  |
| 26 | ` 乌克兰 ` | 105113 |  |
| 27 | ` Latvia ` | 78158 |  |
| 28 | ` Azerbai ` | 64323 |  |
| 29 | ` Bulg ` | 41250 |  |
| 30 | ` Croatian ` | 99070 |  |
| 31 | ` Slovenia ` | 78217 |  |
| 32 | ` Russia ` | 8359 |  |
| 33 | ` Uzbek ` | 89859 |  |
| 34 | ` Sophia ` | 70371 |  |
| 35 | ` Crimea ` | 60711 |  |
| 36 | ` Sweden ` | 96871 |  |
| 37 | ` Catalonia ` | 95852 |  |
| 38 | ` Portuguese ` | 42188 |  |
| 39 | ` Lisbon ` | 80701 |  |
| 40 | ` Hungary ` | 42776 |  |
| 41 | ` Azerbaijan ` | 73477 |  |
| 42 | ` .cz ` | 74504 |  |
| 43 | ` Wrocław ` | 139222 |  |
| 44 | ` Ukr ` | 14299 |  |
| 45 | ` Sevilla ` | 94894 |  |
| 46 | ` Georgian ` | 86442 |  |
| 47 | ` Colombia ` | 38033 |  |
| 48 | ` Poland ` | 27602 |  |
| 49 | ` Укра ` | 127872 |  |
| 50 | ` Copenhagen ` | 63061 |  |
| 51 | ` Bosnia ` | 76328 |  |
| 52 | ` Mongolia ` | 90750 |  |
| 53 | ` Donetsk ` | 97351 |  |
| 54 | ` İstanbul ` | 129514 |  |
| 55 | ` Russian ` | 47707 |  |
| 56 | ` Armenia ` | 71530 |  |
| 57 | ` Belarus ` | 69506 |  |
| 58 | ` Slovak ` | 61264 |  |
| 59 | ` украин ` | 138132 |  |
| 60 | ` Spain ` | 17689 |  |
| 61 | ` Finland ` | 36255 |  |
| 62 | ` Sweden ` | 23190 |  |
| 63 | ` Russians ` | 33966 |  |
| 64 | ` Madrid ` | 24081 |  |
| 65 | ` Peru ` | 47747 |  |
| 66 | ` Russian ` | 8522 |  |
| 67 | ` Honduras ` | 70315 |  |
| 68 | ` Moscow ` | 22415 |  |
| 69 | ` Swedish ` | 30109 |  |
| 70 | ` russian ` | 63145 |  |
| 71 | ` NATO ` | 29234 |  |
| 72 | ` Киев ` | 138868 |  |
| 73 | ` Denmark ` | 34340 |  |
| 74 | ` Helsinki ` | 76334 |  |
| 75 | ` Hungarian ` | 56769 |  |
| 76 | ` Yugosl ` | 74451 |  |
| 77 | ` 莫斯科 ` | 112969 |  |
| 78 | ` Gdańsk ` | 141845 |  |
| 79 | ` Alban ` | 56891 |  |
| 80 | ` Nordic ` | 63366 |  |
| 81 | ` Venezuelan ` | 82649 |  |
| 82 | ` Maced ` | 56452 |  |
| 83 | ` Bolivia ` | 76610 |  |
| 84 | ` Elena ` | 68575 |  |
| 85 | ` Italy ` | 15344 |  |
| 86 | ` sof ` | 85806 |  |
| 87 | ` Istanbul ` | 53368 |  |
| 88 | ` -Russian ` | 84035 |  |
| 89 | ` Spain ` | 74320 |  |
| 90 | ` Venezuela ` | 35903 |  |
| 91 | ` Petersburg ` | 53948 |  |
| 92 | ` Украин ` | 137829 |  |
| 93 | ` cityName ` | 96701 |  |
| 94 | ` 公主 ` | 105454 |  |
| 95 | ` Colombian ` | 83553 |  |
| 96 | ` 葡萄牙 ` | 112334 |  |
| 97 | ` Sof ` | 35547 |  |
| 98 | ` .ua ` | 77695 |  |
| 99 | ` Россия ` | 137421 |  |
| 100 | ` Kremlin ` | 50324 |  |

</details>

### Row 7: Natalie

<details>
<summary>Natalie: predictions 1–100 · target woman · target rank 61 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` actress ` | 23654 |  |
| 2 | ` Natalie ` | 63020 |  |
| 3 | ` Actress ` | 78439 |  |
| 4 | ` singer ` | 22757 |  |
| 5 | ` natal ` | 73339 |  |
| 6 | ` Natal ` | 41601 |  |
| 7 | ` actresses ` | 90028 |  |
| 8 | ` Composer ` | 78771 |  |
| 9 | ` songwriter ` | 84657 |  |
| 10 | ` 她在 ` | 108333 |  |
| 11 | ` she ` | 1340 |  |
| 12 | ` 女主角 ` | 114335 |  |
| 13 | ` 她 ` | 99517 |  |
| 14 | ` dancer ` | 63582 |  |
| 15 | ` Jenner ` | 76064 |  |
| 16 | ` Katie ` | 44463 |  |
| 17 | ` 女星 ` | 114224 |  |
| 18 | ` pornstar ` | 72183 |  |
| 19 | ` 婵 ` | 120821 |  |
| 20 | ` composer ` | 39450 |  |
| 21 | ` girl ` | 3743 |  |
| 22 | ` She ` | 2932 |  |
| 23 | ` Lana ` | 94787 |  |
| 24 | ` vocalist ` | 95565 |  |
| 25 | ` 她是 ` | 107852 |  |
| 26 | ` 龄 ` | 100820 |  |
| 27 | ` Singer ` | 54670 |  |
| 28 | ` 但她 ` | 110204 |  |
| 29 | ` composer ` | 74894 |  |
| 30 | ` girl ` | 28552 |  |
| 31 | ` Chanel ` | 90372 |  |
| 32 | ` Composer ` | 90067 |  |
| 33 | ` 女主 ` | 111938 |  |
| 34 | ` née ` | 130577 |  |
| 35 | ` Utah ` | 22389 |  |
| 36 | ` Kylie ` | 97164 |  |
| 37 | ` herself ` | 11675 |  |
| 38 | ` 年龄 ` | 102185 |  |
| 39 | ` Ventura ` | 84654 |  |
| 40 | ` 歌手 ` | 107434 |  |
| 41 | ` lesbian ` | 24054 |  |
| 42 | ` 女方 ` | 116089 |  |
| 43 | ` .She ` | 98541 |  |
| 44 | ` musician ` | 38744 |  |
| 45 | ` ella ` | 53501 |  |
| 46 | ` Edmonton ` | 45287 |  |
| 47 | ` Sharon ` | 51852 |  |
| 48 | ` 演员 ` | 104534 |  |
| 49 | ` singers ` | 67041 |  |
| 50 | ` brunette ` | 48117 |  |
| 51 | ` pian ` | 59066 |  |
| 52 | ` Wife ` | 42408 |  |
| 53 | ` she ` | 31058 |  |
| 54 | ` Cait ` | 75332 |  |
| 55 | ` 她说 ` | 106095 |  |
| 56 | ` 小姐姐 ` | 117253 |  |
| 57 | ` Colorado ` | 15383 |  |
| 58 | ` heroine ` | 80087 |  |
| 59 | ` Sergei ` | 83998 |  |
| 60 | ` age ` | 4231 |  |
| 61 | ` девуш ` | 137980 |  |
| 62 | ` woman ` | 5220 | ✓ |
| 63 | ` Kardashian ` | 68007 |  |
| 64 | ` Nashville ` | 36540 |  |
| 65 | ` 出生于 ` | 112625 |  |
| 66 | ` Animator ` | 45754 |  |
| 67 | ` Cupertino ` | 96735 |  |
| 68 | ` 好莱 ` | 113475 |  |
| 69 | ` Netflix ` | 22642 |  |
| 70 | ` Girlfriend ` | 98493 |  |
| 71 | ` born ` | 9223 |  |
| 72 | ` célib ` | 94336 |  |
| 73 | ` girlfriend ` | 22761 |  |
| 74 | ` Denver ` | 22117 |  |
| 75 | ` nàng ` | 130453 |  |
| 76 | ` sopr ` | 79129 |  |
| 77 | ` Vanessa ` | 70519 |  |
| 78 | ` 潋 ` | 121506 |  |
| 79 | ` Mormon ` | 36597 |  |
| 80 | ` Female ` | 28638 |  |
| 81 | ` 儋 ` | 121643 |  |
| 82 | ` Ontario ` | 20551 |  |
| 83 | ` 好莱坞 ` | 113476 |  |
| 84 | ` Girl ` | 11363 |  |
| 85 | ` She ` | 7941 |  |
| 86 | ` 西宁 ` | 114690 |  |
| 87 | ` 她的 ` | 104007 |  |
| 88 | ` 范冰冰 ` | 117180 |  |
| 89 | ` guitarist ` | 61640 |  |
| 90 | ` Olivia ` | 59287 |  |
| 91 | ` 蕙 ` | 121562 |  |
| 92 | ` Monterey ` | 97395 |  |
| 93 | ` Duchess ` | 94939 |  |
| 94 | ` Netflix ` | 80466 |  |
| 95 | ` female ` | 8778 |  |
| 96 | ` sisters ` | 29293 |  |
| 97 | ` ovarian ` | 84442 |  |
| 98 | ` Lesbian ` | 52543 |  |
| 99 | ` 跟她 ` | 113647 |  |
| 100 | ` 东莞市 ` | 113960 |  |

</details>

### Row 8: David

<details>
<summary>David: predictions 1–100 · target man · target rank 86,813 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Saul ` | 67524 |  |
| 2 | ` prophet ` | 52504 |  |
| 3 | ` 泰州 ` | 117053 |  |
| 4 | ` novelist ` | 80647 |  |
| 5 | ` theolog ` | 89502 |  |
| 6 | ` .djangoproject ` | 63521 |  |
| 7 | ` Chavez ` | 76241 |  |
| 8 | ` Bolshevik ` | 91401 |  |
| 9 | ` 阿根廷 ` | 109192 |  |
| 10 | ` 首相 ` | 114925 |  |
| 11 | ` Davidson ` | 53245 |  |
| 12 | ` Chile ` | 33000 |  |
| 13 | ` Damascus ` | 57023 |  |
| 14 | ` django ` | 13575 |  |
| 15 | ` 安庆 ` | 118442 |  |
| 16 | ` Zimbabwe ` | 51220 |  |
| 17 | ` bishop ` | 60278 |  |
| 18 | ` 大臣 ` | 111312 |  |
| 19 | ` Prophet ` | 36628 |  |
| 20 | ` biologist ` | 87604 |  |
| 21 | ` astronom ` | 45282 |  |
| 22 | ` instein ` | 36394 |  |
| 23 | ` psychiatrist ` | 72869 |  |
| 24 | ` 舟山 ` | 113409 |  |
| 25 | ` chancellor ` | 76175 |  |
| 26 | ` 瑞典 ` | 110055 |  |
| 27 | ` (actor ` | 65451 |  |
| 28 | ` songwriter ` | 84657 |  |
| 29 | ` 嘉兴 ` | 109575 |  |
| 30 | ` 南非 ` | 113453 |  |
| 31 | ` LDS ` | 71536 |  |
| 32 | ` Auschwitz ` | 92290 |  |
| 33 | ` 渔民 ` | 116613 |  |
| 34 | ` 弑 ` | 120995 |  |
| 35 | ` Nobel ` | 46978 |  |
| 36 | ` Holocaust ` | 43098 |  |
| 37 | ` Palestin ` | 16331 |  |
| 38 | ` lawmaker ` | 70376 |  |
| 39 | ` Rabbi ` | 70869 |  |
| 40 | ` 信阳 ` | 117643 |  |
| 41 | ` guitarist ` | 61640 |  |
| 42 | ` Chúa ` | 132756 |  |
| 43 | ` 霸王 ` | 115560 |  |
| 44 | ` Bashar ` | 70260 |  |
| 45 | ` 鄱 ` | 121470 |  |
| 46 | ` politician ` | 35938 |  |
| 47 | ` 书法家 ` | 117720 |  |
| 48 | ` .djang ` | 63323 |  |
| 49 | ` ólogo ` | 139428 |  |
| 50 | ` 肇庆 ` | 118070 |  |
| 51 | ` 开封 ` | 112491 |  |
| 52 | ` physicist ` | 82223 |  |
| 53 | ` prophets ` | 71668 |  |
| 54 | ` _READONLY ` | 93695 |  |
| 55 | ` LORD ` | 57377 |  |
| 56 | ` (writer ` | 38356 |  |
| 57 | ` Scientist ` | 67309 |  |
| 58 | ` asley ` | 73011 |  |
| 59 | ` (py ` | 46827 |  |
| 60 | ` cháu ` | 136291 |  |
| 61 | ` 绍兴 ` | 109800 |  |
| 62 | ` goddess ` | 51090 |  |
| 63 | ` Psychiat ` | 60664 |  |
| 64 | ` cz ` | 17988 |  |
| 65 | ` Bethlehem ` | 94958 |  |
| 66 | ` 诗人 ` | 106926 |  |
| 67 | ` Cornwall ` | 75950 |  |
| 68 | ` bard ` | 41810 |  |
| 69 | ` martyr ` | 89308 |  |
| 70 | ` Psalm ` | 81876 |  |
| 71 | ` pope ` | 63272 |  |
| 72 | ` 长征 ` | 111159 |  |
| 73 | ` Tanzania ` | 68794 |  |
| 74 | ` saint ` | 35859 |  |
| 75 | ` Zionist ` | 77473 |  |
| 76 | ` 晚年 ` | 114836 |  |
| 77 | ` bdsm ` | 41697 |  |
| 78 | ` 检察官 ` | 117772 |  |
| 79 | ` cuador ` | 49262 |  |
| 80 | ` 南京市 ` | 112891 |  |
| 81 | ` 大庆 ` | 116055 |  |
| 82 | ` poet ` | 39260 |  |
| 83 | ` peasant ` | 88938 |  |
| 84 | ` 抑郁症 ` | 115496 |  |
| 85 | ` Netanyahu ` | 44642 |  |
| 86 | ` urgeon ` | 80462 |  |
| 87 | ` 西湖 ` | 110192 |  |
| 88 | ` psychologist ` | 53495 |  |
| 89 | ` (css ` | 73002 |  |
| 90 | ` disciple ` | 83695 |  |
| 91 | ` 谪 ` | 121327 |  |
| 92 | ` preacher ` | 88753 |  |
| 93 | ` 镇江 ` | 114199 |  |
| 94 | ` Ecuador ` | 53991 |  |
| 95 | ` 嵬 ` | 120947 |  |
| 96 | ` 篙 ` | 121789 |  |
| 97 | ` 诺贝尔 ` | 115832 |  |
| 98 | ` playwright ` | 98368 |  |
| 99 | ` 政协委员 ` | 110933 |  |
| 100 | ` 作家 ` | 105022 |  |

</details>

### Row 9: Dylan

<details>
<summary>Dylan: predictions 1–100 · target man · target rank 83,659 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Dylan ` | 43358 |  |
| 2 | ` songwriter ` | 84657 |  |
| 3 | ` 歌手 ` | 107434 |  |
| 4 | ` singer ` | 22757 |  |
| 5 | ` 诗人 ` | 106926 |  |
| 6 | ` 歌词 ` | 114355 |  |
| 7 | ` Nashville ` | 36540 |  |
| 8 | ` musician ` | 38744 |  |
| 9 | ` Lyrics ` | 56243 |  |
| 10 | ` guitarist ` | 61640 |  |
| 11 | ` 演唱 ` | 103407 |  |
| 12 | ` 摇滚 ` | 117278 |  |
| 13 | ` 歌曲 ` | 106067 |  |
| 14 | ` lyrics ` | 23261 |  |
| 15 | ` lyric ` | 68335 |  |
| 16 | ` _song ` | 57102 |  |
| 17 | ` 唱片 ` | 115142 |  |
| 18 | ` Songs ` | 93156 |  |
| 19 | ` poet ` | 39260 |  |
| 20 | ` vocalist ` | 95565 |  |
| 21 | ` filmmaker ` | 56237 |  |
| 22 | ` Grammy ` | 73579 |  |
| 23 | ` songs ` | 75052 |  |
| 24 | ` Singer ` | 54670 |  |
| 25 | ` songs ` | 11674 |  |
| 26 | ` -song ` | 76640 |  |
| 27 | ` 吉他 ` | 115789 |  |
| 28 | ` Songs ` | 39100 |  |
| 29 | ` 遵义 ` | 116365 |  |
| 30 | ` 这首歌 ` | 117883 |  |
| 31 | ` 乐队 ` | 111818 |  |
| 32 | ` Composer ` | 90067 |  |
| 33 | ` 诺贝尔 ` | 115832 |  |
| 34 | ` 作家 ` | 105022 |  |
| 35 | ` 女星 ` | 114224 |  |
| 36 | ` 演员 ` | 104534 |  |
| 37 | ` (song ` | 60873 |  |
| 38 | ` song ` | 31751 |  |
| 39 | ` poets ` | 72972 |  |
| 40 | ` singers ` | 67041 |  |
| 41 | ` .song ` | 70421 |  |
| 42 | ` Tennessee ` | 23219 |  |
| 43 | ` Harlem ` | 82752 |  |
| 44 | ` 编剧 ` | 111346 |  |
| 45 | ` ylan ` | 36478 |  |
| 46 | ` .actor ` | 41368 |  |
| 47 | ` Belfast ` | 54239 |  |
| 48 | ` Composer ` | 78771 |  |
| 49 | ` .artist ` | 63346 |  |
| 50 | ` drummer ` | 67946 |  |
| 51 | ` Beatles ` | 54857 |  |
| 52 | ` 奥斯卡 ` | 115934 |  |
| 53 | ` Kanye ` | 58717 |  |
| 54 | ` musicians ` | 31529 |  |
| 55 | ` музык ` | 133295 |  |
| 56 | ` Icelandic ` | 99148 |  |
| 57 | ` 专辑 ` | 109468 |  |
| 58 | ` sung ` | 39339 |  |
| 59 | ` _artist ` | 66840 |  |
| 60 | ` .spotify ` | 76453 |  |
| 61 | ` .Actor ` | 76094 |  |
| 62 | ` 巨星 ` | 114447 |  |
| 63 | ` 球星 ` | 112941 |  |
| 64 | ` 射手 ` | 111500 |  |
| 65 | ` 爱尔兰 ` | 116517 |  |
| 66 | ` composer ` | 39450 |  |
| 67 | ` 歌唱 ` | 113375 |  |
| 68 | ` rapper ` | 49337 |  |
| 69 | ` Bowie ` | 67247 |  |
| 70 | ` 一首 ` | 108462 |  |
| 71 | ` rocker ` | 95126 |  |
| 72 | ` Tolkien ` | 79303 |  |
| 73 | ` mús ` | 59857 |  |
| 74 | ` (artist ` | 92880 |  |
| 75 | ` 唱 ` | 100074 |  |
| 76 | ` lyr ` | 85237 |  |
| 77 | ` -po ` | 74388 |  |
| 78 | ` 音乐 ` | 101941 |  |
| 79 | ` spotify ` | 87388 |  |
| 80 | ` composer ` | 74894 |  |
| 81 | ` 艺术家 ` | 106231 |  |
| 82 | ` bard ` | 41810 |  |
| 83 | ` Wyoming ` | 48732 |  |
| 84 | ` 艺人 ` | 108644 |  |
| 85 | ` song ` | 5492 |  |
| 86 | ` filmmakers ` | 62103 |  |
| 87 | ` music ` | 31161 |  |
| 88 | ` bard ` | 65919 |  |
| 89 | ` actor ` | 12089 |  |
| 90 | ` Vanderbilt ` | 83026 |  |
| 91 | ` Broadway ` | 36676 |  |
| 92 | ` _playlist ` | 69267 |  |
| 93 | ` 瀍 ` | 123708 |  |
| 94 | ` Odin ` | 91073 |  |
| 95 | ` Actor ` | 18870 |  |
| 96 | ` 这首 ` | 117882 |  |
| 97 | ` 创作者 ` | 118297 |  |
| 98 | ` actress ` | 23654 |  |
| 99 | ` spotify ` | 87790 |  |
| 100 | ` /music ` | 77770 |  |

</details>

### Row 10: Emily

<details>
<summary>Emily: predictions 1–100 · target woman · target rank 74 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Emily ` | 34166 |  |
| 2 | ` Emily ` | 84048 |  |
| 3 | ` 作家 ` | 105022 |  |
| 4 | ` poet ` | 39260 |  |
| 5 | ` Dickinson ` | 97237 |  |
| 6 | ` 诗人 ` | 106926 |  |
| 7 | ` novelist ` | 80647 |  |
| 8 | ` nàng ` | 130453 |  |
| 9 | ` Hermione ` | 53499 |  |
| 10 | ` hemp ` | 40963 |  |
| 11 | ` 宝妈 ` | 118978 |  |
| 12 | ` 新娘 ` | 111274 |  |
| 13 | ` Ohio ` | 14352 |  |
| 14 | ` heroine ` | 80087 |  |
| 15 | ` actress ` | 23654 |  |
| 16 | ` poems ` | 44219 |  |
| 17 | ` 蛾 ` | 119241 |  |
| 18 | ` 佛山市 ` | 114611 |  |
| 19 | ` writer ` | 6916 |  |
| 20 | ` 她 ` | 99517 |  |
| 21 | ` 蓑 ` | 121132 |  |
| 22 | ` 她是 ` | 107852 |  |
| 23 | ` 作者 ` | 57421 |  |
| 24 | ` 小女孩 ` | 112060 |  |
| 25 | ` 女子 ` | 104340 |  |
| 26 | ` 榆林 ` | 117903 |  |
| 27 | ` 她在 ` | 108333 |  |
| 28 | ` CSP ` | 90510 |  |
| 29 | ` 芙 ` | 103639 |  |
| 30 | ` author ` | 3150 |  |
| 31 | ` ohio ` | 95885 |  |
| 32 | ` .author ` | 16314 |  |
| 33 | ` Female ` | 28638 |  |
| 34 | ` Shelley ` | 90509 |  |
| 35 | ` physicist ` | 82223 |  |
| 36 | ` 懿 ` | 114139 |  |
| 37 | ` Scientist ` | 67309 |  |
| 38 | ` Victorian ` | 43177 |  |
| 39 | ` 闺 ` | 103606 |  |
| 40 | ` 荔 ` | 103777 |  |
| 41 | ` 女主角 ` | 114335 |  |
| 42 | ` 瑶 ` | 106103 |  |
| 43 | ` 顺德 ` | 113019 |  |
| 44 | ` Writer ` | 29404 |  |
| 45 | ` maid ` | 42065 |  |
| 46 | ` 笔 ` | 100041 |  |
| 47 | ` 瑛 ` | 119148 |  |
| 48 | ` Composer ` | 78771 |  |
| 49 | ` 莹 ` | 107911 |  |
| 50 | ` poets ` | 72972 |  |
| 51 | ` feminism ` | 52010 |  |
| 52 | ` Childhood ` | 71310 |  |
| 53 | ` 黛 ` | 112783 |  |
| 54 | ` goddess ` | 51090 |  |
| 55 | ` playwright ` | 98368 |  |
| 56 | ` 女 ` | 57750 |  |
| 57 | ` Girl ` | 11363 |  |
| 58 | ` 婵 ` | 120821 |  |
| 59 | ` Hemp ` | 56208 |  |
| 60 | ` 姑娘 ` | 102373 |  |
| 61 | ` Ohio ` | 65117 |  |
| 62 | ` female ` | 8778 |  |
| 63 | ` 龄 ` | 100820 |  |
| 64 | ` 女孩 ` | 101339 |  |
| 65 | ` blonde ` | 26026 |  |
| 66 | ` Spencer ` | 35121 |  |
| 67 | ` 卉 ` | 103216 |  |
| 68 | ` childhood ` | 19990 |  |
| 69 | ` Literary ` | 73624 |  |
| 70 | ` novels ` | 31863 |  |
| 71 | ` 宝鸡 ` | 116708 |  |
| 72 | ` 妇 ` | 99981 |  |
| 73 | ` girl ` | 3743 |  |
| 74 | ` Heidi ` | 81836 |  |
| 75 | ` woman ` | 5220 | ✓ |
| 76 | ` 怡 ` | 106433 |  |
| 77 | ` songwriter ` | 84657 |  |
| 78 | ` 娥 ` | 116511 |  |
| 79 | ` writer ` | 18189 |  |
| 80 | ` she ` | 31058 |  |
| 81 | ` 苓 ` | 119824 |  |
| 82 | ` .She ` | 98541 |  |
| 83 | ` Danielle ` | 71616 |  |
| 84 | ` therapist ` | 41763 |  |
| 85 | ` 婦 ` | 111917 |  |
| 86 | ` biom ` | 38438 |  |
| 87 | ` niece ` | 79015 |  |
| 88 | ` 但她 ` | 110204 |  |
| 89 | ` 小说 ` | 104032 |  |
| 90 | ` 的女孩 ` | 111374 |  |
| 91 | ` author ` | 3094 |  |
| 92 | ` she ` | 1340 |  |
| 93 | ` Oregon ` | 18784 |  |
| 94 | ` _writer ` | 28908 |  |
| 95 | ` 姊 ` | 103856 |  |
| 96 | ` writer ` | 38959 |  |
| 97 | ` Biom ` | 90508 |  |
| 98 | ` 钗 ` | 119260 |  |
| 99 | ` Cleveland ` | 23475 |  |
| 100 | ` 宝安 ` | 116343 |  |

</details>

### Row 11: Michael

<details>
<summary>Michael: predictions 1–100 · target man · target rank 74,755 (0-based)</summary>

| Prediction rank | J-lens token | Token ID | Exact target |
| ---: | --- | ---: | :---: |
| 1 | ` Psychiat ` | 60664 |  |
| 2 | ` actress ` | 23654 |  |
| 3 | ` psychiatrist ` | 72869 |  |
| 4 | ` ochond ` | 46417 |  |
| 5 | ` actor ` | 12089 |  |
| 6 | ` Michaels ` | 86756 |  |
| 7 | ` actor ` | 93410 |  |
| 8 | ` 霆 ` | 103523 |  |
| 9 | ` Saul ` | 67524 |  |
| 10 | ` actor ` | 5621 |  |
| 11 | ` .actor ` | 41368 |  |
| 12 | ` Actor ` | 18870 |  |
| 13 | ` Actor ` | 24718 |  |
| 14 | ` songwriter ` | 84657 |  |
| 15 | ` (actor ` | 65451 |  |
| 16 | ` composer ` | 74894 |  |
| 17 | ` instein ` | 36394 |  |
| 18 | ` physician ` | 27279 |  |
| 19 | ` ichael ` | 59708 |  |
| 20 | ` .Actor ` | 76094 |  |
| 21 | ` _actor ` | 54818 |  |
| 22 | ` chancellor ` | 76175 |  |
| 23 | ` vocalist ` | 95565 |  |
| 24 | ` guitarist ` | 61640 |  |
| 25 | ` Müller ` | 98918 |  |
| 26 | ` filmmaker ` | 56237 |  |
| 27 | ` Micha ` | 69920 |  |
| 28 | ` biologist ` | 87604 |  |
| 29 | ` NVIDIA ` | 33561 |  |
| 30 | ` 演员 ` | 104534 |  |
| 31 | ` Mormon ` | 36597 |  |
| 32 | ` Giuliani ` | 88050 |  |
| 33 | ` singer ` | 22757 |  |
| 34 | ` 医师 ` | 107158 |  |
| 35 | ` surgeon ` | 44998 |  |
| 36 | ` 丞 ` | 118794 |  |
| 37 | ` urgeon ` | 80462 |  |
| 38 | ` theolog ` | 89502 |  |
| 39 | ` atholic ` | 95330 |  |
| 40 | ` Lawyer ` | 70214 |  |
| 41 | ` lawyer ` | 15417 |  |
| 42 | ` NYPD ` | 73155 |  |
| 43 | ` Composer ` | 90067 |  |
| 44 | ` michael ` | 87906 |  |
| 45 | ` composer ` | 39450 |  |
| 46 | ` Nietzsche ` | 93106 |  |
| 47 | ` Actress ` | 78439 |  |
| 48 | ` 青岛市 ` | 111483 |  |
| 49 | ` physicist ` | 82223 |  |
| 50 | ` chicago ` | 92208 |  |
| 51 | ` Chicago ` | 46175 |  |
| 52 | ` ichten ` | 28698 |  |
| 53 | ` viewHolder ` | 43473 |  |
| 54 | ` 镇江 ` | 114199 |  |
| 55 | ` /MIT ` | 85163 |  |
| 56 | ` dancer ` | 63582 |  |
| 57 | ` sergeant ` | 83121 |  |
| 58 | ` MIC ` | 44741 |  |
| 59 | ` rü ` | 126354 |  |
| 60 | ` McLaren ` | 71364 |  |
| 61 | ` Composer ` | 78771 |  |
| 62 | ` viewController ` | 95412 |  |
| 63 | ` musician ` | 38744 |  |
| 64 | ` Chicago ` | 10557 |  |
| 65 | ` Beatles ` | 54857 |  |
| 66 | ` Spielberg ` | 93616 |  |
| 67 | ` 醫師 ` | 115903 |  |
| 68 | ` rück ` | 129579 |  |
| 69 | ` �� ` | 78910 |  |
| 70 | ` Cuomo ` | 67207 |  |
| 71 | ` viewController ` | 52722 |  |
| 72 | ` senator ` | 31697 |  |
| 73 | ` Paulo ` | 36926 |  |
| 74 | ` 白癜风 ` | 105834 |  |
| 75 | ` Superintendent ` | 67884 |  |
| 76 | ` redhead ` | 91705 |  |
| 77 | ` django ` | 13575 |  |
| 78 | ` Auschwitz ` | 92290 |  |
| 79 | ` 湖人 ` | 107982 |  |
| 80 | ` MIC ` | 54961 |  |
| 81 | ` Wolfgang ` | 86498 |  |
| 82 | ` Mayer ` | 73711 |  |
| 83 | ` wargs ` | 6674 |  |
| 84 | ` Physician ` | 87198 |  |
| 85 | ` (sz ` | 40571 |  |
| 86 | ` fä ` | 131660 |  |
| 87 | ` .toJSON ` | 44150 |  |
| 88 | ` nephew ` | 62804 |  |
| 89 | ` chemist ` | 74088 |  |
| 90 | ` 泰州 ` | 117053 |  |
| 91 | ` éri ` | 27055 |  |
| 92 | ` iegel ` | 74947 |  |
| 93 | ` 鲨 ` | 119494 |  |
| 94 | ` 编剧 ` | 111346 |  |
| 95 | ` therapist ` | 41763 |  |
| 96 | ` divorced ` | 49492 |  |
| 97 | ` fried ` | 97525 |  |
| 98 | ` pha ` | 3967 |  |
| 99 | ` .guard ` | 73529 |  |
| 100 | ` PhD ` | 29561 |  |

</details>

## Provenance and reproduction

[Original Facts about measurement](name_gender_facts_about_measurement.json) · [Top-100 extraction script](export_name_gender_facts_about_top100.py) · [Table rendering script](render_name_gender_facts_about_tables.py)

Inference requires the original measurement workspace’s `jlens.model_sweep` modules and cached checkpoint. The exporter checks hashes for the measurement modules and data, verifies numerical interventions, and verifies equality to the saved Facts about run. The rendering script needs only the committed JSON files and Python’s standard library.

```sh
python code/docs/qwen3-8b-base-no-years/render_name_gender_facts_about_tables.py
```
