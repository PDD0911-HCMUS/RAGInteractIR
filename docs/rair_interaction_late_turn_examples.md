# RAIR-VF Late-Turn Interaction Examples

These examples are selected from real experiment logs. The target is outside top-10 for at least the early interaction turns and enters top-10 only after later user feedback. They are suitable for qualitative visualization of multi-turn interaction.

## CLIP fusion lambda=0.9

### Example 1: image `72258`

- Target path: `train2014/COCO_train2014_000000072258.jpg`
- Target caption: aerial view of 4 toothbrushes in a holder
- Rank timeline across turns: `T0: 13 -> T1: 15 -> T2: 15 -> T3: 15 -> T4: 19 -> T5: 8`
- Key transition: turn 5 moves target rank `19` -> `8`
- User initial input: electric toothbrushes in a metallic holder with reflections

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | electric toothbrushes in a metallic holder with reflections | 13 | initial retrieval only | none |
| 1 | electric toothbrushes in silver holder reflections | 15 | main intent: Refine electric toothbrush search with visual details from the evidence.; ambiguities: material of holder; color of toothbrushes; location of holder; missing: specific brand of toothbrush; style of holder | suggestions: blue and white toothbrus... | action=edit; selected=[blue and white toothbrushes; silver cup holder]; added=[blue and white toothbrushes; silver cup holder]; refined query="electric toothbrushes in a metallic holder with reflections, blue and white toothbrushes, silver cup holder" |
| 2 | electric toothbrushes in silver holder reflections | 15 | main intent: Refine image search for electric toothbrushes in silver holders with reflections.; ambiguities: color of toothbrushes; type of holder; bathroom setting; missing: specific brand of toothbrush; style of holder; overall bathroom aesthetic | sugges... | action=edit; selected=[clear and blue toothbrushes]; added=[the toothbrushes have reflections]; refined query="electric toothbrushes in a metallic holder with reflections, blue and white toothbrushes, silver cup holder, the toothbrushes have reflections" |
| 3 | electric toothbrushes in silver holder reflections | 15 | main intent: Refine image search for electric toothbrushes in silver holders with reflections.; ambiguities: color of toothbrushes; type of holder; bathroom setting; missing: specific brand of toothbrush; style of holder; overall bathroom aesthetic | sugges... | action=edit; selected=[clear and blue toothbrushes]; added=[the toothbrushes have reflections]; refined query="electric toothbrushes in a metallic holder with reflections, blue and white toothbrushes, silver cup holder, the toothbrushes have reflections" |
| 4 | electric toothbrushes in metallic holder reflections | 19 | main intent: Refine image search for electric toothbrushes in silver holders with reflections.; ambiguities: color of toothbrushes; type of holder; bathroom setting; missing: specific brand of toothbrush; style of holder; overall bathroom aesthetic | sugges... | action=edit; selected=[clear and blue toothbrushes]; added=[the toothbrushes are lined up on the counter]; removed/rejected=[white sink; chrome faucet]; refined query="electric toothbrushes in a metallic holder with reflections, blue and white toothbrushes,... |
| 5 | electric toothbrushes metallic holder reflections | 8 | main intent: Refine image search for electric toothbrushes in metallic holders with reflections.; ambiguities: type of holder (cup, caddy, base); color of toothbrushes/holder; bathroom setting; missing: specific brand of toothbrush or toothpaste; style of b... | action=edit; selected=[blue and white electric toothbrushes; dark grey toothbrush holder; electric base]; added=[blue and white electric toothbrushes; dark grey toothbrush holder; electric base]; removed/rejected=[chrome faucet; white sink]; refined query="... |

### Example 2: image `110493`

- Target path: `train2014/COCO_train2014_000000110493.jpg`
- Target caption: an open-face egg salad and onion sandwich with a whipped cream dessert on the side
- Rank timeline across turns: `T0: 41 -> T1: 81 -> T2: 11 -> T3: 11 -> T4: 10 -> T5: 10`
- Key transition: turn 4 moves target rank `11` -> `10`
- User initial input: close-up of food with herring, mashed potatoes, and whipped cream

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | close-up of food with herring, mashed potatoes, and whipped cream | 41 | initial retrieval only | none |
| 1 | herring mashed potatoes whipped cream close-up | 81 | main intent: Refine image search for a close-up of food with specific ingredients.; ambiguities: type of food; style of presentation; missing: specific type of herring; style of mashed potatoes; type of whipped cream | suggestions: white square plate; food ... | action=edit; selected=[white square plate]; added=[a white square plate]; refined query="close-up of food with herring, mashed potatoes, and whipped cream on a white square plate" |
| 2 | herring mashed potatoes whipped cream | 11 | main intent: Find images of mashed potatoes with whipped cream and herring.; ambiguities: herring presence; specific vegetables; missing: plating style; overall dish presentation | suggestions: grilled salmon mashed potatoes vegetables; white plate mashed p... | action=edit; selected=[white plate mashed potatoes]; added=[a white square plate]; refined query="herring mashed potatoes whipped cream close-up on a white square plate" |
| 3 | herring mashed potatoes whipped cream | 11 | main intent: Refine image search for a dish containing mashed potatoes, whipped cream, and herring.; ambiguities: type of meat; type of vegetables; style of presentation; missing: specific type of herring; style of whipped cream; plating style | suggestions... | action=edit; selected=[white square plate]; added=[a white square plate]; refined query="herring mashed potatoes whipped cream close-up on a white square plate" |
| 4 | herring mashed potatoes whipped cream white plate | 10 | main intent: Refine image search for a dish containing mashed potatoes, whipped cream, and herring.; ambiguities: type of meat; type of vegetables; style of presentation; missing: specific type of herring; style of whipped cream; plating style | suggestions... | action=edit; selected=[white square plate]; added=[a white square plate]; refined query="herring mashed potatoes whipped cream on a white square plate" |
| 5 | herring mashed potatoes whipped cream white plate | 10 | initial retrieval only | action=stop_found |

### Example 3: image `338319`

- Target path: `train2014/COCO_train2014_000000338319.jpg`
- Target caption: a man tossing a frisbee in a park with green grass
- Rank timeline across turns: `T0: 19 -> T1: 57 -> T2: 57 -> T3: 57 -> T4: 6 -> T5: 6`
- Key transition: turn 4 moves target rank `57` -> `6`
- User initial input: man swinging baseball bat in park

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | man swinging baseball bat in park | 19 | initial retrieval only | none |
| 1 | man swinging baseball bat park | 57 | suggestions: metal baseball bat; man in gray shirt; baseball field | action=edit; selected=[metal baseball bat]; added=[metal baseball bat]; refined query="man swinging metal baseball bat in park" |
| 2 | man swinging baseball bat park | 57 | main intent: Refine image search for a man swinging a baseball bat in a park-like setting.; ambiguities: age of the man; type of bat (metal or wood); professional or friendly game; presence of other players/umpires; missing: specific clothing; field conditi... | action=edit; selected=[light yellow bat]; added=[light yellow bat]; refined query="man swinging light yellow baseball bat in park" |
| 3 | man swinging baseball bat park | 57 | main intent: Refine image search for a man swinging a baseball bat in a park-like setting.; ambiguities: age of the man; type of bat (metal or wood); professional or friendly game; presence of other players/umpires; missing: specific clothing; field conditi... | action=edit; selected=[light yellow bat]; added=[light yellow bat]; refined query="man swinging light yellow baseball bat in park" |
| 4 | man swinging baseball bat park light yellow | 6 | main intent: Refine image search for a man swinging a baseball bat in a park-like setting.; ambiguities: age of the man; type of bat (metal or wood); professional or friendly game; presence of other players/umpires; missing: specific clothing; field conditi... | action=edit; selected=[light yellow bat]; added=[light yellow bat]; refined query="man swinging light yellow baseball bat in park" |
| 5 | man swinging baseball bat park light yellow | 6 | initial retrieval only | action=stop_found |

### Example 4: image `571146`

- Target path: `train2014/COCO_train2014_000000571146.jpg`
- Target caption: a woman holding a nintendo wii game controller
- Rank timeline across turns: `T0: 56 -> T1: 71 -> T2: 59 -> T3: 38 -> T4: 8 -> T5: 8`
- Key transition: turn 4 moves target rank `38` -> `8`
- User initial input: person playing video game with Wii remote in living room

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | person playing video game with Wii remote in living room | 56 | initial retrieval only | none |
| 1 | person playing video game Wii remote | 71 | main intent: Refine image search for a person playing video games with a Wii remote in a living room.; ambiguities: age of person; number of people; visibility of TV; specific furniture; missing: specific game; details of the living room | suggestions: whit... | action=edit; selected=[white Wii remote]; added=[white Wii remote]; refined query="person playing video game with white Wii remote in living room" |
| 2 | person playing Wii remote | 59 | main intent: Refine image search for a person playing a video game with a Wii remote.; ambiguities: gender of person; setting; missing: specific hand; details of the person's appearance | suggestions: white Wii remote; Wii remote with nunchuks; hand holding... | action=edit; selected=[white Wii remote]; added=[white Wii remote]; refined query="person playing video game Wii remote with white Wii remote in living room" |
| 3 | person playing Wii remote standing | 38 | main intent: Refine image search for a person playing a Wii remote.; ambiguities: gender of person; specific hand; missing: setting; activity beyond holding the remote | suggestions: white Wii remote; Wii remote with nunchuks; person holding Wii remote on c... | action=edit; selected=[white Wii remote]; added=[the person is standing]; removed/rejected=[the person is on a couch]; refined query="person playing Wii remote while standing" |
| 4 | woman standing playing Wii remote | 8 | suggestions: white Wii remote; person playing Wii Fit; person standing on balance board | action=edit; selected=[person playing Wii remote standing]; added=[the person is female]; removed/rejected=[the person is on a couch]; refined query="person playing Wii remote standing (female)" |
| 5 | woman standing playing Wii remote | 8 | initial retrieval only | action=stop_found |

### Example 5: image `176617`

- Target path: `train2014/COCO_train2014_000000176617.jpg`
- Target caption: there is a man that is holding up 2 frisbees
- Rank timeline across turns: `T0: 63 -> T1: 12 -> T2: 12 -> T3: 7 -> T4: 7`
- Key transition: turn 3 moves target rank `12` -> `7`
- User initial input: man holding two frisbees, wearing sunglasses, outdoor setting

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | man holding two frisbees, wearing sunglasses, outdoor | 63 | initial retrieval only | none |
| 1 | man holding yellow frisbees sunglasses outdoor | 12 | main intent: Refine image search for a man with frisbees.; ambiguities: color of frisbee; clothing details; age of man; missing: specific type of hat; specific park scene | suggestions: yellow frisbee; striped shirt; wearing a hat | action=edit; selected=[yellow frisbee]; added=[the frisbee is yellow]; refined query="man holding two frisbees, wearing sunglasses, outdoor, the frisbee is yellow" |
| 2 | man holding two yellow frisbees sunglasses gray t-shirt outdoor | 12 | main intent: Refine image search for a man holding a yellow frisbee outdoors.; ambiguities: age of man; specific clothing details; exact outdoor location; missing: specific activity beyond holding the frisbee; style of sunglasses | suggestions: man red t-sh... | action=edit; selected=[man red t-shirt frisbee; man sunglasses park; man short hair frisbee]; added=[the frisbee is yellow]; removed/rejected=[man wearing a red t-shirt; man wearing a headband]; refined query="man holding two frisbees, wearing sunglasses, o... |
| 3 | man holding yellow frisbees sunglasses | 7 | main intent: Refine image search for a man holding frisbees.; ambiguities: color of t-shirt; age of man; missing: specific activity (throwing, catching, spinning) | suggestions: man red t-shirt; man white frisbee; man 20s | action=edit; selected=[man red t-shirt]; added=[man wearing a gray t-shirt]; removed/rejected=[man wearing a red t-shirt]; refined query="man holding two yellow frisbees sunglasses gray t-shirt outdoor" |
| 4 | man holding yellow frisbees sunglasses | 7 | initial retrieval only | action=stop_found |

## SigLIP fusion lambda=0.9

### Example 1: image `478331`

- Target path: `train2014/COCO_train2014_000000478331.jpg`
- Target caption: a skiier poses in front of a ski lift and dark clouds
- Rank timeline across turns: `T0: not found -> T1: not found -> T2: 80 -> T3: 80 -> T4: 80 -> T5: 3`
- Key transition: turn 5 moves target rank `80` -> `3`
- User initial input: person skiing on snowy slope with ski lift in background

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | person skiing on snowy slope with ski lift | not found | initial retrieval only | none |
| 1 | person skiing snowy slope ski lift red jacket helmet | not found | main intent: Refine image search for a person skiing on a snowy slope with a ski lift.; ambiguities: gender of person; ski type (skis vs. snowboard); clothing color; missing: specific clothing details; viewpoint (e.g., close-up, wide shot) | suggestions: pe... | action=edit; selected=[person wearing a red jacket; person wearing a helmet]; added=[person wearing a red jacket; person wearing a helmet]; refined query="person skiing on snowy slope with ski lift, wearing a red jacket and a helmet" |
| 2 | person skiing snowy slope ski lift red jacket helmet goggles | 80 | main intent: Refine image search for a person skiing.; ambiguities: type of skiing (downhill, cross-country); age/gender of person; snow conditions; missing: specific type of slope (steep, crest); details of the helmet | suggestions: person skiing black ski... | action=edit; selected=[person skiing with ski poles]; added=[person wearing goggles]; refined query="person skiing snowy slope ski lift red jacket helmet goggles" |
| 3 | person skiing snowy slope ski lift red jacket helmet goggles | 80 | main intent: Refine image search for a person skiing.; ambiguities: type of skiing (downhill, cross-country); ski color; background details; missing: ski pole presence; specific goggle color | suggestions: person skiing black skis; person skiing black helme... | action=edit; selected=[person skiing black skis; person skiing black helmet; person skiing pinkish goggles]; added=[person wearing a red jacket; person wearing a helmet; person wearing goggles]; refined query="person skiing snowy slope ski lift red jacket h... |
| 4 | person skiing snowy slope ski lift red jacket helmet goggles | 80 | main intent: Refine image search for a person skiing.; ambiguities: type of skiing (downhill, cross-country); ski color; background details; missing: ski pole presence; specific goggle color | suggestions: person skiing black skis; person skiing black helme... | action=edit; selected=[person skiing black skis; person skiing black helmet; person skiing pinkish goggles]; added=[person wearing a red jacket; person wearing a helmet; person wearing goggles]; refined query="person skiing snowy slope ski lift red jacket h... |
| 5 | person skiing snowy slope ski lift helmet goggles black jacket | 3 | main intent: Refine image search for a person skiing.; ambiguities: type of skiing (downhill, cross-country); ski color; background details; missing: ski pole presence; specific goggle color | suggestions: person skiing black skis; person skiing black helme... | action=edit; selected=[person skiing black skis; person skiing black helmet; person skiing pinkish goggles]; removed/rejected=[person wearing a red jacket]; refined query="person skiing snowy slope ski lift black jacket helmet goggles" |

### Example 2: image `90925`

- Target path: `train2014/COCO_train2014_000000090925.jpg`
- Target caption: a pink, orange, yellow and green kite on the ground at a park
- Rank timeline across turns: `T0: 29 -> T1: 17 -> T2: 20 -> T3: 20 -> T4: 20 -> T5: 9`
- Key transition: turn 5 moves target rank `20` -> `9`
- User initial input: man flying colorful kite in field at sunset

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | man flying colorful kite in field at sunset | 29 | initial retrieval only | none |
| 1 | man flying rainbow kite in field sunset | 17 | main intent: Refine image search for a man flying a colorful kite at sunset.; ambiguities: age of the man; specific colors of the kite; details of the field; missing: specific type of kite; clothing of the man | suggestions: rainbow kite; man wearing a hat;... | action=edit; selected=[rainbow kite]; added=[rainbow kite]; refined query="man flying rainbow kite in field at sunset" |
| 2 | man flying rainbow kite field sunset | 20 | main intent: Refine image search for a man flying a rainbow kite in a field at sunset.; ambiguities: age of man; specific colors of kite; type of field; missing: specific details of the sunset; man's attire | action=edit; selected=[the boy is playing with his colorful kite; a person flies a kite with streamers in a field; a person is in a field attempting to fly a kite]; added=[the kite is rainbow colored]; removed/rejected=[the boy is playing with his colorful ... |
| 3 | man flying rainbow kite field sunset | 20 | main intent: Refine image search for a man flying a rainbow kite in a field at sunset.; ambiguities: age of man; details of the field; specific colors of the kite; missing: specific time of sunset; presence of other people | suggestions: red and yellow kite... | action=edit; selected=[a person is in a field attempting to fly a kite]; added=[the kite is rainbow colored]; removed/rejected=[the boy is playing with his colorful kite; the field overgrown is slightly overgrown; you see buildings; answer: off in the dista... |
| 4 | man flying rainbow kite field sunset | 20 | main intent: Refine image search for a man flying a rainbow kite in a field at sunset.; ambiguities: age of man; details of the field; specific colors of the kite; missing: specific time of sunset; presence of other people | suggestions: red and yellow kite... | action=edit; selected=[a person is in a field attempting to fly a kite]; added=[the kite is rainbow colored]; removed/rejected=[the boy is playing with his colorful kite; the field overgrown is slightly overgrown; you see buildings; answer: off in the dista... |
| 5 | man flying rainbow kite grassy field sunset | 9 | main intent: Refine image search for a man flying a rainbow kite in a field at sunset.; ambiguities: age of man; details of the field; specific colors of the kite; missing: specific time of sunset; presence of other people | suggestions: red and yellow kite... | action=edit; selected=[red and yellow kite; grassy field]; added=[the kite has red and yellow sections; the field is grassy]; refined query="man flying rainbow kite with red and yellow sections in a grassy field at sunset" |

### Example 3: image `257847`

- Target path: `train2014/COCO_train2014_000000257847.jpg`
- Target caption: a zebra walking through the tall grass in front of some trees
- Rank timeline across turns: `T0: 14 -> T1: 23 -> T2: 84 -> T3: 84 -> T4: 8 -> T5: 8`
- Key transition: turn 4 moves target rank `84` -> `8`
- User initial input: zebra standing in tall grass with trees in background

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | zebra standing in tall grass with trees | 14 | initial retrieval only | none |
| 1 | zebra in tall grass with trees | 23 | main intent: Find an image of a zebra in a grassy environment with trees.; ambiguities: color of grass; age of zebra; presence of other animals; missing: specific tree type; zebra activity | suggestions: brown grass and trees; striped zebra; zebra with gree... | action=edit; selected=[brown grass and trees; striped zebra; zebra with green trees]; added=[the zebra has stripes]; refined query="zebra standing in tall grass with trees and stripes" |
| 2 | zebra in tall green grass with trees | 84 | main intent: Refine image search for a zebra in tall grass with trees.; ambiguities: age of zebra; density of trees; color of grass; missing: specific type of trees; time of day | suggestions: zebra in green grass; zebra with trees and shrubs; zebra in tall... | action=edit; selected=[zebra in green grass; zebra with trees and shrubs; zebra in tall grass with leaves on trees]; added=[the grass is green]; refined query="zebra standing in tall grass with trees and green grass" |
| 3 | zebra in tall green grass with trees | 84 | main intent: Find an image of a zebra in a grassy environment with trees.; ambiguities: tall grass vs. short grass; number of zebras; type of trees; missing: specific tree type; time of year (leaves on trees) | suggestions: zebra grazing | action=edit; selected=[a zebra standing in a field of grass outside; a zebra in a field of grass with a tree near by; a zebra grazing on a grass covered field]; added=[the grass is tall and green; there are trees in sight]; removed/rejected=[the grass short... |
| 4 | zebra standing in tall green grass, trees, bushes | 8 | main intent: Find an image of a zebra in a grassy environment with trees.; ambiguities: tall grass vs. short grass; number of zebras; type of trees; missing: specific tree type; time of year (leaves on trees) | suggestions: zebra grazing | action=edit; selected=[zebra grazing]; added=[the zebra is not grazing]; removed/rejected=[the zebra is grazing]; refined query="zebra standing in tall green grass with trees" |
| 5 | zebra standing in tall green grass, trees, bushes | 8 | initial retrieval only | action=stop_found |

### Example 4: image `111549`

- Target path: `train2014/COCO_train2014_000000111549.jpg`
- Target caption: an image of a plate of food on a white plate
- Rank timeline across turns: `T0: 11 -> T1: 11 -> T2: 11 -> T3: 10 -> T4: 10`
- Key transition: turn 3 moves target rank `11` -> `10`
- User initial input: plate with rice, pickled vegetables, and meat slices

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | plate with rice, pickled vegetables, and meat slices | 11 | initial retrieval only | none |
| 1 | plate rice pickled carrots zucchini meat | 11 | suggestions: white rice; carrots and zucchini; steak and turkey | action=edit; selected=[white rice; carrots and zucchini; steak and turkey]; added=[pickled vegetables include carrots and zucchini]; removed/rejected=[sauce with the vegetables; answer: it looks like soy sauce; what is the meat: looks like chicken maybe; wh... |
| 2 | plate rice pickled carrots zucchini meat | 11 | main intent: Refine image search for a plate of food.; ambiguities: type of meat; type of rice; style of cooking; missing: specific cut of meat; preparation method of vegetables; overall presentation style | suggestions: white rice; sliced carrots; coated meat | action=edit; selected=[white rice; sliced carrots]; removed/rejected=[coated meat]; refined query="plate with rice, pickled carrots and zucchini, sliced meat" |
| 3 | plate rice pickled carrots zucchini coated meat | 10 | main intent: Refine image search for a plate of food.; ambiguities: type of meat; type of rice; style of cooking; missing: specific cut of meat; preparation method of vegetables; overall presentation style | suggestions: white rice; sliced carrots; coated meat | action=edit; selected=[white rice; sliced carrots; coated meat]; refined query="plate with rice, pickled vegetables (carrots and zucchini), and coated meat" |
| 4 | plate rice pickled carrots zucchini coated meat | 10 | initial retrieval only | action=stop_found |

### Example 5: image `174680`

- Target path: `train2014/COCO_train2014_000000174680.jpg`
- Target caption: a man holding a nintendo wii game controller
- Rank timeline across turns: `T0: 15 -> T1: 11 -> T2: 27 -> T3: 8 -> T4: 8`
- Key transition: turn 3 moves target rank `27` -> `8`
- User initial input: man playing video game with Wii remote

| Turn | RAIR query | Target rank | RAIR response / suggestions | User edit/selection |
|---:|---|---:|---|---|
| 0 | man playing video game with Wii remote | 15 | initial retrieval only | none |
| 1 | man playing Wii with remote | 11 | main intent: Refine image search for a man playing video games with a Wii remote.; ambiguities: age of the man; setting (couch, room); game genre; missing: specific game being played; man's activity (sitting, lying) | suggestions: man with glasses playing W... | action=edit; selected=[man with glasses playing Wii]; added=[man with glasses playing Wii]; refined query="man playing video game with Wii remote and glasses" |
| 2 | man playing Wii with remote glasses | 27 | main intent: Refine image search for a man playing Wii with a remote.; ambiguities: age of man; room setting; game being played; missing: specific game; room details; man's attire | suggestions: man wearing glasses; man wearing short sleeve shirt; playing i... | action=edit; selected=[man wearing glasses]; added=[man wearing glasses]; refined query="man playing Wii with remote and wearing glasses" |
| 3 | man playing Wii with white remote | 8 | main intent: Refine image search for a man playing Wii with remote glasses.; ambiguities: age of man; setting; game being played; missing: specific clothing details; facial expression | suggestions: man wearing black metal glasses; man wearing a white shirt... | action=edit; selected=[man wearing black metal glasses; man wearing a white shirt; white Wii remote]; added=[man wearing glasses; man wearing a white shirt; white Wii remote]; refined query="man playing Wii with remote glasses" |
| 4 | man playing Wii with white remote | 8 | initial retrieval only | action=stop_found |

