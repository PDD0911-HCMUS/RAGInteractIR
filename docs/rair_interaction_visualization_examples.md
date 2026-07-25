# RAIR-VF Interaction Visualization Candidates

Selection criterion: RAIR has non-empty suggestions; user performs an active edit/selection; target is outside top-10 or not found before the interaction, then enters top-10 after the feedback is applied. These examples are for qualitative visualization, not aggregate evaluation.

## CLIP fusion lambda=0.9

Candidates matching criterion: 75. Selected examples: 5. Skipped malformed/error samples: 69.

### Example 1: image `324705`

- Target path: `train2014/COCO_train2014_000000324705.jpg`
- Target caption: people watching a man give a glass blowing demonstration
- Transition: turn 2 moves target rank `not found` -> `1`
- Rank timeline across turns: `T0: 92 -> T1: not found -> T2: 1 -> T3: 1`
- User initial input: industrial workshop with glassblowing equipment and people
- RAIR current query before feedback: industrial workshop glassblowing equipment
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine image search for industrial workshop with glassblowing equipment.
- RAIR suggestions shown to user:
  - stainless steel hot plate (add_detail) - Specifies a visible appliance color and type.
  - hand saws and tools (add_detail) - Highlights the presence of tools observed in the evidence.
  - refrigerator and shop equipment (add_detail) - Combines two key elements found in the workshop setting.
- User edit/selection:
  - action: `edit`
  - selected suggestions: stainless steel hot plate; hand saws and tools; refrigerator and shop equipment
  - kept constraints: industrial workshop glassblowing equipment; metal pipes workshop
  - added constraints: people present
  - added target details: people present
  - user refined query: industrial workshop glassblowing equipment people
  - reason: The current suggestions do not fully capture the presence of people in the industrial workshop, which is a key element of the target image.
- RAIR query after receiving feedback: industrial glassblowing workshop
- Target rank after feedback retrieval: `1`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000324705.jpg` score=0.3591 | people watching a man give a glass blowing demonstration <- target
  - #2: `train2014/COCO_train2014_000000086676.jpg` score=0.3395 | female cook seems mesmerized by the huge flames in the pot
  - #3: `train2014/COCO_train2014_000000378416.jpg` score=0.3367 | a outdoor kitchen, pots and pans hanging from the ceiling
  - #4: `train2014/COCO_train2014_000000063636.jpg` score=0.3361 | a man putting something into a small fireplace
  - #5: `val2014/COCO_val2014_000000486290.jpg` score=0.3324 | a big group of glass cups that are different colors
  - #6: `train2014/COCO_train2014_000000236375.jpg` score=0.3319 | a man standing in front of a fire oven
  - #7: `val2014/COCO_val2014_000000172342.jpg` score=0.3290 | patrons sitting at bar of seafood restaurant being served by employee behind the bar
  - #8: `val2014/COCO_val2014_000000401212.jpg` score=0.3258 | a large building has a glass business logo on it
  - #9: `train2014/COCO_train2014_000000404254.jpg` score=0.3253 | a bunch of utensils are sitting near a sink
  - #10: `train2014/COCO_train2014_000000151566.jpg` score=0.3248 | the man is using the pottery wheel to make something

### Example 2: image `90311`

- Target path: `train2014/COCO_train2014_000000090311.jpg`
- Target caption: a tennis player is running on a tennis court
- Transition: turn 1 moves target rank `not found` -> `1`
- Rank timeline across turns: `T0: not found -> T1: 1 -> T2: 1`
- User initial input: {"initial_query":"tennis player on court with scoreboard showing '10'}
- RAIR current query before feedback: tennis player on court scoreboard showing 10
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine image search for a tennis player on a court with a scoreboard showing 10.
- RAIR suggestions shown to user:
  - green tennis court (add_detail) - Multiple captions mention a green court.
  - black and orange racket (add_detail) - Several captions describe a racket with black and orange colors.
  - white and black tennis dress (add_detail) - One caption specifically mentions a white and black tennis dress.
- User edit/selection:
  - action: `edit`
  - selected suggestions: green tennis court
  - kept constraints: tennis player on court scoreboard showing 10
  - added constraints: green tennis court
  - added target details: green tennis court
  - user refined query: tennis player on a green court scoreboard showing 10
- RAIR query after receiving feedback: tennis player green court scoreboard 10
- Target rank after feedback retrieval: `1`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000090311.jpg` score=0.3635 | a tennis player is running on a tennis court <- target
  - #2: `val2014/COCO_val2014_000000476460.jpg` score=0.3604 | the score keeper at a tennis match turns the numbers
  - #3: `train2014/COCO_train2014_000000168187.jpg` score=0.3598 | a man jumping to hit a tennis ball back to his opponent
  - #4: `val2014/COCO_val2014_000000427901.jpg` score=0.3598 | playoff match in tennis doubles is being played
  - #5: `train2014/COCO_train2014_000000431566.jpg` score=0.3579 | a tennis player hitting the ball with her tennis racket
  - #6: `train2014/COCO_train2014_000000295575.jpg` score=0.3578 | kids and adults playing on a tennis court
  - #7: `val2014/COCO_val2014_000000346934.jpg` score=0.3556 | a tennis player getting ready to hit a serve
  - #8: `val2014/COCO_val2014_000000046101.jpg` score=0.3530 | a game is being played on the green court
  - #9: `train2014/COCO_train2014_000000144353.jpg` score=0.3514 | 2 people on tennis court with crowd of people watching
  - #10: `train2014/COCO_train2014_000000022811.jpg` score=0.3493 | a woman playing tennis and preparing to hit the ball

### Example 3: image `271418`

- Target path: `train2014/COCO_train2014_000000271418.jpg`
- Target caption: a small bird with a yellow body sitting on a tree branch
- Transition: turn 1 moves target rank `not found` -> `4`
- Rank timeline across turns: `T0: not found -> T1: 4 -> T2: 4`
- User initial input: bird perched on tree branch with green foliage background
- RAIR current query before feedback: bird perched on tree branch with green foliage
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine image search for a bird on a branch with foliage.
- RAIR suggestions shown to user:
  - bird with long beak (add_detail) - One caption specifically mentions a long beak.
  - yellow and black bird (add_detail) - One caption describes the bird as yellow and black.
- User edit/selection:
  - action: `edit`
  - selected suggestions: yellow and black bird
  - kept constraints: bird perched on tree branch with green foliage
  - added constraints: yellow and black bird
  - added target details: yellow and black bird
  - user refined query: bird perched on tree branch with green foliage, yellow and black bird
- RAIR query after receiving feedback: yellow and black bird on tree branch
- Target rank after feedback retrieval: `4`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000556683.jpg` score=0.4076 | a yellow and black bird is sitting on a branch
  - #2: `train2014/COCO_train2014_000000280740.jpg` score=0.3794 | a bird perched on a branch with leaves
  - #3: `train2014/COCO_train2014_000000086576.jpg` score=0.3787 | a yellow bird sitting on a branch eating food
  - #4: `train2014/COCO_train2014_000000271418.jpg` score=0.3778 | a small bird with a yellow body sitting on a tree branch <- target
  - #5: `val2014/COCO_val2014_000000452224.jpg` score=0.3728 | a brightly colored bird perches on a tree branch
  - #6: `train2014/COCO_train2014_000000529902.jpg` score=0.3722 | a bird is sitting on the limb of a tree
  - #7: `val2014/COCO_val2014_000000108353.jpg` score=0.3712 | yellow bird staring off in distance on tree branch
  - #8: `val2014/COCO_val2014_000000222407.jpg` score=0.3705 | a bird with blue eyes and a yellow tail standing on a branch
  - #9: `train2014/COCO_train2014_000000173864.jpg` score=0.3703 | a small yellow bird perched on top of a tree branch
  - #10: `train2014/COCO_train2014_000000366455.jpg` score=0.3702 | a bird sitting on top of a large tree branch

### Example 4: image `334302`

- Target path: `train2014/COCO_train2014_000000334302.jpg`
- Target caption: computer keyboard and address book on cluttered desk
- Transition: turn 1 moves target rank `not found` -> `5`
- Rank timeline across turns: `T0: not found -> T1: 5 -> T2: 5`
- User initial input: desk with keyboard, notebook, lighter, and glass
- RAIR current query before feedback: desk with keyboard, notebook, lighter, and glass
- RAIR current target rank: `not found`
- RAIR suggestions shown to user:
  - desk with black keyboard (add_detail) - Several images show a black keyboard.
  - desk with light brown wood (add_detail) - Some images depict a light brown wooden desk.
  - desk with silver laptop (add_detail) - One image shows a silver laptop.
- User edit/selection:
  - action: `edit`
  - selected suggestions: desk with black keyboard; desk with light brown wood; desk with silver laptop
  - kept constraints: desk with keyboard, notebook, lighter, and glass
  - added constraints: black keyboard; light brown wood; silver laptop
  - added target details: black keyboard; light brown wood; silver laptop
  - user refined query: desk with black keyboard, light brown wood, silver laptop, notebook, lighter, and glass
- RAIR query after receiving feedback: desk keyboard notebook lighter glass
- Target rank after feedback retrieval: `5`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000010907.jpg` score=0.3596 | the sun shines on a keyboard through a window
  - #2: `val2014/COCO_val2014_000000173532.jpg` score=0.3561 | a computer keyboard sitting next to a mouse
  - #3: `train2014/COCO_train2014_000000128074.jpg` score=0.3540 | a white mouse and keyboard sit together on a desk
  - #4: `train2014/COCO_train2014_000000025193.jpg` score=0.3535 | a desktop computer keyboard sitting on a desk
  - #5: `train2014/COCO_train2014_000000334302.jpg` score=0.3481 | computer keyboard and address book on cluttered desk <- target
  - #6: `val2014/COCO_val2014_000000297426.jpg` score=0.3459 | 4 photographs of the portions of a keyboard and a mouse
  - #7: `train2014/COCO_train2014_000000053385.jpg` score=0.3448 | there is a mac brand mouse and keyboard
  - #8: `train2014/COCO_train2014_000000073226.jpg` score=0.3429 | 2 laptops behind a keyboard on a desktop
  - #9: `train2014/COCO_train2014_000000054892.jpg` score=0.3411 | a close up of a mouse with a keyboard
  - #10: `val2014/COCO_val2014_000000116252.jpg` score=0.3403 | a small tablet attached to a key board, with a mouse

### Example 5: image `173196`

- Target path: `train2014/COCO_train2014_000000173196.jpg`
- Target caption: young girls on grassy field with animal during competition
- Transition: turn 1 moves target rank `not found` -> `6`
- Rank timeline across turns: `T0: not found -> T1: 6 -> T2: 6`
- User initial input: two people riding sheep in field with helmets
- RAIR current query before feedback: two people riding sheep in field with helmets
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine image search for two people riding sheep with helmets.
- RAIR suggestions shown to user:
  - white sheep in field (add_detail) - Several candidates mention white sheep.
  - people in a grassy field (add_detail) - Reinforces the field setting.
  - old photograph (add_detail) - Several candidates describe the image as an old photograph.
- User edit/selection:
  - action: `edit`
  - selected suggestions: white sheep in field; people in a grassy field; old photograph
  - kept constraints: two people riding sheep in field with helmets
  - added constraints: the sheep are black
  - added target details: black sheep
  - removed details: white sheep; field overgrown is well kept; old photograph
  - user refined query: two people riding black sheep in field with helmets
- RAIR query after receiving feedback: two people riding black sheep in field with helmets
- Target rank after feedback retrieval: `6`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000014683.jpg` score=0.3863 | 2 people ride on horseback through a grassy field
  - #2: `train2014/COCO_train2014_000000348058.jpg` score=0.3760 | the cowboy is concerned about the safety of the sheep
  - #3: `train2014/COCO_train2014_000000309526.jpg` score=0.3605 | a man riding a bike with his helmet on through the grass
  - #4: `train2014/COCO_train2014_000000251475.jpg` score=0.3577 | a person feeding a sheep behind a fence
  - #5: `val2014/COCO_val2014_000000234057.jpg` score=0.3572 | a sheep standing with a lamb in the grass
  - #6: `train2014/COCO_train2014_000000173196.jpg` score=0.3559 | young girls on grassy field with animal during competition <- target
  - #7: `train2014/COCO_train2014_000000188976.jpg` score=0.3536 | 2 sheep and 2 babies are grazing in the field
  - #8: `train2014/COCO_train2014_000000424138.jpg` score=0.3506 | black cow looks at black dog in grassy field
  - #9: `train2014/COCO_train2014_000000274685.jpg` score=0.3503 | 2 men riding horses in a grassy setting
  - #10: `val2014/COCO_val2014_000000066561.jpg` score=0.3469 | a couple of sheep are in a grassy field

## SigLIP fusion lambda=0.9

Candidates matching criterion: 52. Selected examples: 5. Skipped malformed/error samples: 47.

### Example 1: image `481893`

- Target path: `train2014/COCO_train2014_000000481893.jpg`
- Target caption: a double decker bus traveling past a horse statue
- Transition: turn 1 moves target rank `not found` -> `2`
- Rank timeline across turns: `T0: not found -> T1: 2 -> T2: 2`
- User initial input: red double-decker bus with 'King's Cross' sign, London
- RAIR current query before feedback: red double-decker bus 'King's Cross' London
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine image search for a red double-decker bus at King's Cross in London.
- RAIR suggestions shown to user:
  - red double-decker bus route number (add_detail) - Several images show a route number on the bus.
  - red double-decker bus at night (add_detail) - Some images depict the scene at night with lights on.
  - red and black double-decker bus (add_detail) - Some images show the bus has black accents.
- User edit/selection:
  - action: `edit`
  - selected suggestions: red double-decker bus route number
  - kept constraints: red double-decker bus 'King's Cross' London
  - added constraints: route number 10
  - added target details: route number 10
  - user refined query: red double-decker bus 'King's Cross' London with route number 10
- RAIR query after receiving feedback: red double-decker bus King's Cross London route 10
- Target rank after feedback retrieval: `2`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000143401.jpg` score=0.2173 | red double-decker bus coming to a stop at a city
  - #2: `train2014/COCO_train2014_000000481893.jpg` score=0.2018 | a double decker bus traveling past a horse statue <- target
  - #3: `train2014/COCO_train2014_000000175943.jpg` score=0.1949 | a double-decker red bus on a london street
  - #4: `train2014/COCO_train2014_000000283679.jpg` score=0.1915 | a red double decker bus driving down a street
  - #5: `train2014/COCO_train2014_000000462057.jpg` score=0.1869 | red double decker bus parked in tall garage
  - #6: `val2014/COCO_val2014_000000319655.jpg` score=0.1833 | a double decker bus on a london street at night
  - #7: `val2014/COCO_val2014_000000073256.jpg` score=0.1822 | red london buses driving down a city street
  - #8: `train2014/COCO_train2014_000000456042.jpg` score=0.1809 | a red double decker bus pulls out of a bus terminal
  - #9: `train2014/COCO_train2014_000000264151.jpg` score=0.1800 | a red double-decker bus parked next to a sidewalk
  - #10: `val2014/COCO_val2014_000000199492.jpg` score=0.1798 | there is a red double decker bus at the side of the street

### Example 2: image `561015`

- Target path: `train2014/COCO_train2014_000000561015.jpg`
- Target caption: a young boy riding on top of a skateboard
- Transition: turn 1 moves target rank `not found` -> `3`
- Rank timeline across turns: `T0: not found -> T1: 3 -> T2: 3`
- User initial input: child wearing helmet skateboarding on concrete path
- RAIR current query before feedback: child wearing helmet skateboarding on concrete path
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine image search for a child skateboarding with a helmet.
- RAIR suggestions shown to user:
  - blue helmet (add_detail) - Multiple captions mention a blue helmet.
  - shorts and t shirt (add_detail) - Several captions describe the child wearing shorts and a t-shirt.
  - gray sidewalk (add_detail) - Multiple captions mention a gray sidewalk.
- User edit/selection:
  - action: `edit`
  - selected suggestions: blue helmet
  - kept constraints: child wearing helmet skateboarding on concrete path
  - added constraints: blue helmet
  - added target details: blue helmet
  - user refined query: child wearing blue helmet skateboarding on concrete path
- RAIR query after receiving feedback: child skateboarding blue helmet concrete
- Target rank after feedback retrieval: `3`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000443913.jpg` score=0.2213 | a kid in blue shirt with helmet stepping on a skateboard
  - #2: `val2014/COCO_val2014_000000384723.jpg` score=0.2022 | a young boy tries out skateboarding tricks on a road
  - #3: `train2014/COCO_train2014_000000561015.jpg` score=0.1920 | a young boy riding on top of a skateboard <- target
  - #4: `train2014/COCO_train2014_000000461595.jpg` score=0.1913 | a toddler with a helmet on gets on a skateboard
  - #5: `train2014/COCO_train2014_000000255889.jpg` score=0.1893 | a young skateboarder wearing safety equipment skateboarding down a sidewalk
  - #6: `train2014/COCO_train2014_000000458925.jpg` score=0.1893 | a young boy wearing a helmet skateboards down the sidewalk
  - #7: `train2014/COCO_train2014_000000410739.jpg` score=0.1861 | a small child holding to a cement wall while using a skateboard
  - #8: `train2014/COCO_train2014_000000434326.jpg` score=0.1861 | a boy standing on a porch with a helmet on while holding a skateboard
  - #9: `train2014/COCO_train2014_000000069081.jpg` score=0.1848 | an abstract photograph of a man dressed all in blue skateboarding
  - #10: `train2014/COCO_train2014_000000107628.jpg` score=0.1832 | a boy with a helmet riding on a skateboard

### Example 3: image `167892`

- Target path: `train2014/COCO_train2014_000000167892.jpg`
- Target caption: a red and yellow train traveling down train tracks under a building
- Transition: turn 1 moves target rank `not found` -> `6`
- Rank timeline across turns: `T0: not found -> T1: 6 -> T2: 6`
- User initial input: train at station platform with brick building overhead
- RAIR current query before feedback: train at station platform with brick building
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine image search for a train at a station platform with a brick building.
- RAIR suggestions shown to user:
  - yellow and green train (add_detail) - Several candidates mention a yellow and green train.
  - brown and yellow train (add_detail) - Some candidates describe a brown and yellow train.
  - modern train station (add_detail) - One candidate describes a modern, European-style station.
- User edit/selection:
  - action: `edit`
  - selected suggestions: brown and yellow train
  - kept constraints: train at station platform with brick building
  - added constraints: the train is brown and yellow
  - added target details: the train is brown and yellow
  - user refined query: train at station platform with brick building, brown and yellow
- RAIR query after receiving feedback: brown and yellow train on brick station platform
- Target rank after feedback retrieval: `6`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000247110.jpg` score=0.1806 | a yellow railroad train pulling into the station
  - #2: `train2014/COCO_train2014_000000043506.jpg` score=0.1706 | a yellow train is coming down some tracks
  - #3: `val2014/COCO_val2014_000000190546.jpg` score=0.1676 | a yellow train traveling down tracks next to a train station
  - #4: `val2014/COCO_val2014_000000350966.jpg` score=0.1664 | a subway train pulling into the train station
  - #5: `train2014/COCO_train2014_000000430191.jpg` score=0.1646 | a yellow and silver train stopped at a train station
  - #6: `train2014/COCO_train2014_000000167892.jpg` score=0.1619 | a red and yellow train traveling down train tracks under a building <- target
  - #7: `val2014/COCO_val2014_000000096488.jpg` score=0.1615 | a maroon and yellow train waits at a platform
  - #8: `train2014/COCO_train2014_000000309024.jpg` score=0.1609 | a yellow train passing by another train next to buildings
  - #9: `train2014/COCO_train2014_000000259527.jpg` score=0.1603 | a yellow passenger train sitting on a track underneat a bridge
  - #10: `train2014/COCO_train2014_000000456520.jpg` score=0.1601 | an old train sits at a train station platform

### Example 4: image `530925`

- Target path: `train2014/COCO_train2014_000000530925.jpg`
- Target caption: a baseball pitcher in a red uniform has just let 1 fly
- Transition: turn 2 moves target rank `not found` -> `7`
- Rank timeline across turns: `T0: not found -> T1: not found -> T2: 7 -> T3: 7`
- User initial input: baseball player pitching on mound
- RAIR current query before feedback: baseball player pitching
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine baseball player pitching image search.
- RAIR suggestions shown to user:
  - red and white baseball uniform (add_detail) - Specifies the uniform color combination seen in multiple captions.
  - baseball glove (add_detail) - Highlights a common and visually distinct element.
  - pitcher on mound (add_detail) - Clarifies the location of the player.
- User edit/selection:
  - action: `edit`
  - selected suggestions: a baseball pitcher getting ready to throw a ball
  - kept constraints: baseball player pitching on mound; pitcher in windup phase of pitch
  - added constraints: pitcher wearing a red and white uniform
  - added target details: red and white baseball uniform
  - user refined query: baseball player pitching on mound, pitcher wearing a red and white uniform
- RAIR query after receiving feedback: baseball player pitching red and white uniform
- Target rank after feedback retrieval: `7`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000313313.jpg` score=0.2100 | a baseball player follows through after launching the ball
  - #2: `train2014/COCO_train2014_000000295804.jpg` score=0.2089 | man in red about to pitch a ball
  - #3: `train2014/COCO_train2014_000000350657.jpg` score=0.2007 | pitcher in red and grey uniform throwing a baseball
  - #4: `val2014/COCO_val2014_000000567304.jpg` score=0.1943 | a baseball player throwing a pitch from the mound
  - #5: `train2014/COCO_train2014_000000011737.jpg` score=0.1941 | a baseball player attempting to throw a ball
  - #6: `train2014/COCO_train2014_000000463649.jpg` score=0.1939 | a man pitching a baseball on a baseball field
  - #7: `train2014/COCO_train2014_000000530925.jpg` score=0.1936 | a baseball pitcher in a red uniform has just let 1 fly <- target
  - #8: `train2014/COCO_train2014_000000127268.jpg` score=0.1935 | a young baseball player in red prepares a pitch
  - #9: `train2014/COCO_train2014_000000547726.jpg` score=0.1930 | a boy throwing a baseball wearing a ball glove
  - #10: `train2014/COCO_train2014_000000575119.jpg` score=0.1928 | a baseball player getting ready to throw the ball

### Example 5: image `304548`

- Target path: `train2014/COCO_train2014_000000304548.jpg`
- Target caption: a male snowboarder wearing sunglasses and winter attire snowboards down an incline
- Transition: turn 3 moves target rank `not found` -> `7`
- Rank timeline across turns: `T0: 43 -> T1: 81 -> T2: not found -> T3: 7 -> T4: 7`
- User initial input: person snowboarding in snowy forest landscape
- RAIR current query before feedback: snowboarding in snowy forest, orange and green snowboard
- RAIR current target rank: `not found`
- RAIR diagnosis summary: Refine image search for snowboarding in a snowy forest with a specific snowboard color combination.
- RAIR suggestions shown to user:
  - snowboarder wearing a hat (add_detail) - Adds a detail about the snowboarder's attire, supported by multiple captions.
- User edit/selection:
  - action: `edit`
  - selected suggestions: snowboarder wearing a hat
  - kept constraints: person snowboarding in snowy forest landscape; the snowboard is orange; the snowboard is green
  - added constraints: the snowboarder is wearing sunglasses
  - added target details: the snowboarder is wearing sunglasses
  - user refined query: snowboarding in snowy forest, orange and green snowboard, snowboarder wearing sunglasses
- RAIR query after receiving feedback: snowboarding in snowy forest, orange and green snowboard, sunglasses
- Target rank after feedback retrieval: `7`
- Top retrieved images after feedback:
  - #1: `train2014/COCO_train2014_000000254241.jpg` score=0.1684 | snowboarder performing trick on snow with trees in background
  - #2: `val2014/COCO_val2014_000000214447.jpg` score=0.1684 | someone is snowboarding in the woods with fresh snow
  - #3: `val2014/COCO_val2014_000000376246.jpg` score=0.1605 | snowboarder in a red jacket moving near a row of trees
  - #4: `train2014/COCO_train2014_000000250518.jpg` score=0.1543 | the snowboarder is enjoying h time on the mountainis
  - #5: `train2014/COCO_train2014_000000409025.jpg` score=0.1535 | man adjusting his boot straps on his snowboard
  - #6: `train2014/COCO_train2014_000000271490.jpg` score=0.1528 | a person is snowboarding while snow is falling
  - #7: `train2014/COCO_train2014_000000304548.jpg` score=0.1524 | a male snowboarder wearing sunglasses and winter attire snowboards down an incline <- target
  - #8: `val2014/COCO_val2014_000000351793.jpg` score=0.1516 | a man who is in some water on his snowboard
  - #9: `train2014/COCO_train2014_000000177163.jpg` score=0.1511 | a man snow boarding in a bunch of snow with surrounding trees
  - #10: `val2014/COCO_val2014_000000200563.jpg` score=0.1511 | a person rides down a mountain on a snowboard
