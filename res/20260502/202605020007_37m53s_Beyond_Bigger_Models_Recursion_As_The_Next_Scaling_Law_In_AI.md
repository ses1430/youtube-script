---
title: Beyond Bigger Models: Recursion As The Next Scaling Law In AI
uploader: Y Combinator
channel: Y Combinator
channel_url: https://www.youtube.com/channel/UCcefcZRL2oaA_uBNeo5UOWg
duration: 2273
upload_date: 20260501
webpage_url: https://www.youtube.com/watch?v=DGtUUMNYLcc
id: DGtUUMNYLcc
categories:
  - Science & Technology
tags:
  - YC
  - Y Combinator
---

# Beyond Bigger Models: Recursion As The Next Scaling Law In AI

Welcome back to another episode of Decoded.
Today, I'm back with YC visiting partner Francois Chauvard
to talk about one of the most interesting recent trends in AI research, recursion.
Specifically, we're going to talk about how we can improve a model's reasoning performance
by using recursion at inference time, rather than by just making the model bigger and bigger.
There were two papers that made the power of this approach really clear in 2025.
One on hierarchical reasoning models, or HRM, and another on tiny recursive models, TRM.
Francois, thanks for joining us.
Can you tell us a little bit about these two models, and what was so interesting about them?
Sure.
I guess to set up a little bit of a foundation, you already did an amazing lecture on RNNs and LLMs
in one of the previous videos, so I won't overdo it, but just to give the cliff notes.
An RNN is just a model that you recursively call again and again and again on itself,
and we were very much in the belief that this was required to get to AGI peak RNN use,
which was probably until 2016 with Alex Graves NeurIPS keynote, which is just fantastic,
and all his adaptive compute time work.
So this is about 10 years ago.
People were working on these models.
This was in the era of LSDMs and LSDMs with attention.
Yeah, and depending on which professors you talked to before attention was invented.
And I think what really was the limiting step on RNNs in general
was this thing called backprop through time, where you have to roll out the model,
and then to update the weights, you need to approximate the gradient,
and you step back, back, back, and you keep rolling out.
And as the model gets bigger and bigger, and as you roll out for more and more steps,
then you have all these accumulation of errors, and the gradient gets nosier and nosier,
and then it just kind of stops to work.
Yeah, so you have these like vanishing or exploding gradient problems,
and that's because if you have an input with 20 steps,
you're like multiplying these matrices 20 times, and that causes training.
And we're talking about doing context length of like a million or like a billion,
and so like it's not even just 20, it's like a billion.
And even worse, you have to retain the activations at every single step.
And so like if this were happening in your brain,
you would need like a million copies of your brain at every single activation
so that I can backprop through it.
There's tricks around this that you can do,
and you can do a gradient checkpointing and things like that to reduce that issue,
but then you're just like trading off memory for wall clock time and compute.
Right, so now if you could cross that with LLMs,
the ones that people are widely using,
these while at face value they appear to be similar,
at training time they're doing basically this one-shot feed-forward process
for every input, right?
The LLM, the transformer block, can take all of the inputs in parallel.
It's not actually iteratively going over them one at a time at train time,
so you don't have this needing-to-store-tons-of-activations problem
or this giant vanishing gradients problem with them.
Yeah, exactly.
It's actually all happening in time in one shot magically,
and that was like the trill or lower triangle trick that kind of happens,
this causal mask that occurs,
and so you actually do all time steps in one shot,
and you forward pass a feed-forward model on all time steps in one shot,
and you backwards in one shot,
and it's amazing for train time in terms of like wall clock.
It requires a lot of flops,
and it still requires a lot of the memory.
You still need it there,
but you don't have the vanishing gradient issue,
and what you actually paid for that you have to give up
is this latent reasoning thing
and this compression in the time direction.
There is no compression in LLMs.
Every single decode that I do,
I still have to retain the entire Shakespeare novel
just to like decode a little bit,
and in RNNs, you don't have to do that.
It's all compressed in this hidden state
that you kind of roll out.
Okay, so let's talk about that in a little bit more detail.
Like, you refer to this inherent reasoning ability.
You know, many people think about LLMs as doing reasoning,
and we're going to talk about that a little bit later,
but help me understand where you see the biggest limitations
in LLMs reasoning ability
is in terms of what the model does
in an actual forward pass.
Yeah, and so I guess we go back to chat GPT-2.
GPT-2 was this landmark architecture and paper
that basically was just next token, next token, next token,
and it kind of worked,
and like we just watched val loss go down,
perplexity goes down.
Like, the model just is more performant.
It looks better.
It starts to make some Shakespeare
that actually sounds somewhat plausible,
and then we have to get these things to reason,
and to actually solve some really hard problems,
and I've done extensive experiments on this,
but like if you take, for example, sort,
you have infinite amounts of unsorted lists,
and you give it sorted lists.
You keep feeding it to the model.
It should work, right?
It's actually impossible for the model
to map from unsorted lists to sorted lists.
In a one-shot, basically.
In a one-shot basis.
It's like literally that we know
a theoretical lower bound that for comparison sort,
you can't do better than n log n steps,
and if I have a list that's 31 characters or elements long,
and my transformer is 30,
I run out of steps to do comparisons.
It's not possible for me to like do all the steps
that is needed to be done.
In HRM and TRM, they use Sudoku as an incompressible problem.
Similarly, and so are mazes.
Those are incompressible problems.
Rolling sum, incompressible problem.
So when you mention the sorting algorithm,
when I think back to my algorithms class from college,
the one way you could get faster than n log n in a sorting algorithm
is if you had some access to an external memory cache.
If you had some tape you could write to,
then you can actually do faster than n log n
by basically selectively putting things onto this memory.
And I suspect that's, you know, a key limitation of these LLMs
in that because there's no external memory tape inbuilt into the model,
you lose certain performance possibilities in terms of how fast you can go.
That's right.
And so I guess like radix sort would be like the most common one.
You had like depending on the number of buckets that you have,
you can kind of get from n log n to order n.
You can't get less than n.
You have to touch all the elements.
Sorry, you have to do that.
And if you run out of layers and transformer layers in your neural network,
then you ran out of chances to do that.
Yeah, so this is just like a,
this is like going back to like Alan Turing now and like a Turing machine, right?
Like what's the analogy there exactly that we should think about in terms of LLMs,
I guess not quite satisfying how you think about a Turing machine.
Yeah, so if we, let's just talk about like chat GBT2,
GBT2, the original like no bells and whistles.
It's just a feed forward model.
And so it's just forward passing one step and-
Taking an input, creating a bunch of outputs.
In the Sudoku case, if I have 50 different squares and it's provable that I can only do one
given this information and I have this many layers, then that's all I can do.
And the cheat is the chain of thought.
And so it's completely true that at test time, they are Turing complete
and you can simulate all turn computable functions at test time.
But how do you get it to learn it?
You need to train it.
And that's where, unless you're training it on human labeled traces,
for which there's a lot of problems like the millennial prize problem,
we don't have the trace for it.
Right.
So we'd love to have the trace for it.
It doesn't exist.
Totally.
Makes sense.
Okay.
So with that context in mind now, let's talk about these two papers.
Because I think that sets up a lot of the contrast we're going to draw between these papers
and the models that people are maybe more used to.
So let's talk about HRMs first.
Walk me through a little bit about how this model works and some of the intuition behind it.
Sure.
So this is directly in the lineage of RNNs.
There's not that much novel from like the RNN standpoint, at least in my opinion.
They do have this idea of, you know, from inspired by the brain where I have like,
there's different parts of the brain that operate on different frequencies.
There's some that operate at a really high frequency, which is on the low level of the hierarchy,
some that operate in a really low frequency, which is the higher level of the hierarchy.
And the interplay between those things is really interesting.
So this is like literally in the human brain, there's some like bio inspiration here,
which is that like you have like different waves running at different frequencies
in different parts of the brain or something like that.
Okay, cool.
And I guess that's one interpretation of it, of the way that they're talking about,
you know, classifying these hierarchies of frequencies.
And the most interesting part, at least for me, is the way that they train the neural network.
You take in some X, some input, whether it's an incomplete Sudoku puzzle, a maze, or an art prize challenge.
You do TL steps with the lower level module.
Then you do, to go to H, you do that TH times, and then you have NSUP outer refinement steps.
Yeah, so you basically are like running through the input with a given matrix,
with a given transformation repeatedly on it.
And you're doing that through two levels of refinement and then basically running that process several times.
Yes, so there's exactly three levels of recursion occurring here.
There's the low level, there's the high level, and then there's the outer refinement steps.
And we're calling it recursion because it's the same weights that are being applied repeatedly.
We're not changing the weights in between these steps.
Exactly right.
You get to recurse on the Lnet, TL times.
You've recurse on the TH and the TL, this looped recursion, TH times.
And then you do NSUP, you do this whole outer refinement step, NSUP times.
Cool.
And so, what's the basic intuition for why that works?
Like, why does that produce an effective paper result?
And what even were the results that this paper showed?
Yeah, and so, I mean, this got state of the art on ArcPrize 1 and 2.
This was only a 27 million parameter model that was only trained on ArcPrize.
So it's like 1,000 inputs or something like that, like puzzles basically.
Literally 1,000 tasks, which is extremely small.
There is no pre-training at all.
This starts from like literally Tagula Rasa weights.
And it can outperform at that time if we go back.
You know, we had O3, if you remember back, way back when.
And it, O3 gets zero, literally zero.
And this got like something like 70% on ArcPrize 1 at least at the time,
which was just a huge breakthrough.
And so kind of the way you can kind of think this is like variable scoping.
And so like if I have like, you know, three nested functions,
I guess the first, the lowest level function has like scoped variables,
which they'll call ZL, which is the carry that inits the zero.
A latent variable.
Latent variable, and like traditional RNN literature,
they would call this the hidden state, the low-level hidden state.
And I get to recurse, recurse, recurse, and then I pass back that ZL
back to the outer scoped function, the higher level one.
I let that one do one iter.
It goes back and calls the lower level again.
It does this whole thing in a third outer loop, which is called the outer refinement step.
Okay, but when you describe it like that, it seems like it would have the same backprop
through time problem that you would have at our own ends.
And I think they came up with a clever trick to basically get around that.
So like what was that trick that they figured out?
And this is really the crux of the paper that like differentiates it,
in my opinion, in the literature, is they, instead of doing what Alex Graves did in all of his papers,
from neural turning machines to adaptive compute time to differential neural computers,
is he always back propped through all of the recursion steps.
And he was limited by back propped through time.
So you can only make the model so big.
You have all these issues, vanishing gradients, et cetera, et cetera.
And what they do is they kind of have this DEQ of method of doing fixed point iteration.
Sounds like deep equilibrium.
Yeah, deep equilibrium learning.
Where if I take a batch, and this is completely counterintuitive as a computer vision person,
because you'd never do this, but it actually does make sense.
And I'll explain why.
If I take a batch of like ImageNet or CIFR10, and I forward pass through the model,
and I get some loss, and I backprop, and I update the weights,
I would go get a different batch for the next one.
But what they do instead is they actually do that 16 times.
And as you do that, you actually can see the change in your residuals get less and less and less.
And why it actually makes sense is because when, in the RNN case, the ZL and the ZH, which are the carry,
the task carry, start out as-
Like the hidden states.
Yeah.
The hidden states.
Start out at zeroes.
Those are zeroes.
Then we go through this whole loopy recursion, at least the two loops, the two lower loops,
the TL and TH steps.
And then I backprop just through the two modules, just once, and I don't recurse all the way back.
I do a stop grad, I stop right there.
And then there's a huge residual, and then I don't reset ZL and ZH.
I do it again at a different point in the carry or hidden variable space.
And so one can actually look at it as like a different batch every time,
even though it's the same exact X's.
Yeah.
Like the way I kind of think about it is like the 16 or whatever that you're recursing over,
it's like constructing a mini-batch, not from different inputs, but from like different memory
states basically.
It's like across this hidden or carry memory access basically.
And that math holds and it works.
It follows DEQ directly in the event that the ZL and the delta and ZL and the delta and ZH
go to zero, which it actually doesn't do.
And so we'll get to TRM, but Alexia basically shows that it's just not the case,
and you can't actually apply this math.
And that's why it's working.
That's not sufficient support for why it's working.
We actually don't know why it's really working.
And she figures out that you actually can back prop through all the way to the deep
recursion, which we're going to get into TRM in a second.
And that actually improves performance much, much more.
Interesting.
Okay.
So before we get into TRM, yeah, on, you know, on this paper, you know,
I think there's a bunch of different ways people have looked at this, right, in terms of
how they came up with it and then why this may or may not be working.
One, it's a sort of bio-plausibility argument.
As you know, I'm usually not super keen on these.
You know, I think machine learning tends to have a long history of people starting with
bio-plausible arguments and then realizing that there's some variant of them that seems
highly bio-implausible that actually works better.
I think you have examples.
Yeah, classic.
The first deep learning paper that started this whole craziness is AlexNet.
And in AlexNet, there's actually this funny little thing called, like,
local receptive activation or depression or something like that where, like,
once this activation fires, then, like, I have this, like, you know,
over factory region or something like that.
It actually doesn't work at all.
And, like, it didn't work and you didn't need that.
And then VGG came out and said, get rid of all that.
Just go deeper and three-by-three conv and it actually just, like, outperforms dramatically.
And so, like, this is, like, always the case.
Maybe you need to do it to get accepted into neuro-ps.
Yeah, sure, totally, totally, yeah.
You're definitely the expert here,
but what do you consider to be bio-plausible and what's not?
Well, I think that a lot of machine learning literature has
overlapped a lot with people working in neuroscience.
I think it is very natural for us to ask questions about how does our brain work,
because our brain is, like, an incredible instrument that does a ton of computing,
obviously, and does it in a very shockingly efficient manner, it seems like.
And so, a lot of machine learning research has, for a long time, sought analog from how we think
to understand our brain to work and try to encode that in various machine learning systems.
So, from the very basic concept of what a neural network is, it's called a neural network
because we think it's some basic model for what a neuron is,
how certain activation functions work are meant to be inspired by certain biological premises.
Do you think that's a big number?
The thing about them is that often we use bio-plausibility to inspire us to come up with ideas.
Mm-hm.
But we end up veering away from the bio-plausible to something adjacent to them that
is likely bio-implausible, but that seems to work better.
Something that runs better on a GPU.
Exactly. It runs better on a GPU, it's more efficient in some capacity that is relevant
to how we actually encode it in a computational system.
So, I find thinking about bio-plausibility fun and interesting,
and it's definitely a great way to inspire us to think about new things.
But I tend to not be bounded by bio-plausibility when I think about what machine learning systems we
should prioritize working on or think as particularly exciting other than as, you know,
an interesting scientific launching point for a deeper exploration.
I think the version of this that I find more compelling is actually that original discussion
we were having around automata theory, basically.
And honestly, just actually like fundamental data structures and algorithms theory,
which is that if you're running a complex algorithm, having access to sort of a memory
cache is actually very useful for being able to run that algorithm efficiently.
And I kind of think of this set of hidden states or carry as akin to a Turing machine tape
or akin to the radix sort memory bank where you can basically train a model to use this memory
cache in an intelligent way in a single forward pass so that you can get a more efficient time
operation that would otherwise require some sort of more complicated reasoning.
Yeah, I think that a point I wanted to make earlier is that like we did this COT stuff
and this tool use thing as ways to get beyond the limitations of GPT-2.
And so the way that we get, you can actually, I've done this experiment, you can actually,
if you give me infinite amounts of unsorted lists and sorted lists, if I can do chain of thought
and I can do every single step and teach it to do every single step,
then I can actually get it to do sort and become a Turing machine at test time.
And similarly, an even cheaper one that is much easier to do is you teach it and you say,
hey, there's this Python function called sort.
Just call the function.
And like that's the easiest thing to do and you don't need backprop at all.
And so those are the two hacks.
Now, well Francois, this is solved.
Like we're done, right?
No, because I needed to know what sort was.
What happens if we didn't know what merge sort?
The chain of thought is not going to inherently discover sorting from first principles.
It's finding it from our historical knowledge of everything it's trained on.
Yeah.
I mean, this is like the, the Demis had this whole thing about like the ultimate test is the Einstein test.
Like go back to 1911 and then like have it rebuild all the physics up until now.
Similarly, let's just pretend that we only had bubble sort.
We knew other, no other sort system.
If you chain of thought it on all the bubble sort input and output, it will only do bubble sort.
In fact, it won't even do bubble sort that well.
So this is the best situation.
And then the tool use, of course, it can only know bubble sort.
I want to get to merge sort.
How do I discover merge sort?
And I think the interesting thing just to emphasize here,
because it may not have been extremely clear, is there already exists some type of recursion
that people are used to in LLMs, which is chain of thought we mentioned earlier.
But that is a recursion that's happening in the token space of the model's outputs,
not inherent to the model itself.
That's sort of the fundamental limitation is that the model can only do
a feed forward one shot output.
And then we basically just have this hack that if you keep letting it output things,
then it can read its outputs and do somewhat intelligent seeming things with it.
But it seems to sort of be upper bounded by the data that we feed it that, you know,
the labs are very hungrily buying right now.
Yeah.
And not the sort of like inherent underlying recursive reasoning.
Yeah.
So in both cases, both hacks to solve this in COT and tool use,
you're bounded by the bounds of human knowledge.
In the event it's outside the set of human knowledge, then like you're kind of SOL.
And so that's one.
The other, you make a great point about discrete versus latent space.
Reasoning in a discrete, it can only output the carry in the case of LLens,
has to be snapped back to some discrete token space.
And in the case of RNNs in general, they remain in this continuous latent space,
which is much higher dimensional.
If you give me like a tape that's this long and you cut it up into 10 buckets,
like versus all the possible values.
Right, exactly, yeah.
It's much more expressive to be in continuous space.
But we can't train it that way because we actually, you know,
because you're inhibited by backdrop through time largely.
And this is why this paper is so exciting.
Okay, so before we then go over to the TRM paper,
let's just summarize here.
What matters most from the HRM paper that we should take away
before we transition and contrast it with the TRM paper?
Yeah, I think that the number one piece to take away is this outer refinement loop.
The outer refinement loop scales and there's a great breakdown.
Basically, the Sapien authors, which is huge kudos for this paper,
because there's so many innovations in this paper,
didn't really do like a scaling ablations on every single one of the inputs.
But this guy, Constantine at François Chalet's company, India, actually did.
And it's this amazing breakdown that he posted on YouTube that you can go check out.
But basically, the main takeaway is that the outer refinement loops is the main
beneficiary, is the main reason why these things work so well, which Alexia basically
takes the, she found, I think, in parallel and scales up and shows that you can get rid of a lot
of all this other stuff.
Okay, so like a lot of machine learning, the follow-on paper is basically delete 75% of the first paper,
as we've often done in videos here, and keep the magic basically.
Yeah.
So okay, so what's the magic then?
Like what's the part that actually matters in terms of what stays in the TORM paper?
And let's not contrast the core architectural differences between these two papers.
Yeah.
So I think that, I guess if I break it down into two major things, this outer refinement loop thing
is really great and works really well, and that this like truncated backprop through time,
which is backprop through time, except I truncate at some time.
Some earlier point.
Earlier point.
Yeah.
Called T, T back.
T equals one is actually is completely sufficient.
And so truncated backprop through time, T equals one, completely sufficient.
And that's very counterintuitive.
Which is what HRM found.
Which HRM found, and TRM does a little bit further, rather than going through just one
call to the H net and the L net, it actually goes through one full recursion loop.
So if I do it 16 times, I just go back through one time, and that is kind of sufficient.
And if you do it with this like fixed point iteration thing, pseudo fixed point iteration
thing, where you keep hitting it with a gradient at every single step, it like weirdly works.
And this batch size across the carry space, like actually works.
So that part is also kept between these two models.
Yeah.
It seemed like another thing that changed was having these, this sort of double layer of like,
you know, higher order thinking and lower order thinking.
It seems like it collapsed that down into just a single one.
What's the intuition there?
And how does that actually work in the TRM paper?
Yeah.
So it's interesting.
She actually ablates having two separate networks versus just having one.
I guess the more important space is the variable scope.
Is that you should have low level features and high level features.
But the same network.
And so the best performing model.
The same network can extract both basically.
Yeah.
You weight share between the L net and the H net.
It's just called net.
And you do just one transformer layer versus the four like they do in Sapien.
And just whittle it down to one and do more of a cursion.
And that, but you keep ZL and ZH to be distinct and separate.
And she calls it X and Y which I found very confusing.
Yeah.
X, Y, Z.
It's just very confusing and it's just like ZH and ZL is just cleaner.
So if you read the paper, Y is actually like latent space.
Yeah.
It's like Z basically.
And it is not a label.
Yeah, okay.
Which really threw me through.
Whatever, yeah.
But anyway, so I will go through some code here and I'll walk you through it.
So I've replaced all of her nomenclature and used the Sapien notation,
which is much cleaner and more straightforward to me at least.
Okay, cool.
And now before we dive into the code for a sec,
like in terms of how these TRMs actually work,
it's pretty interesting because this recursion advantage now gives you a bunch of advantages over
transformers where rather than having 500 or a thousand or a million or whatever transformer layers
and having tons and tons of parameters, you get compute depth basically without this parameter depth.
And the optimization process looks like more of like an iterative kind of like expectation
maximization algorithm.
Do you want to talk about how that worked in the TRM paper?
Because I thought that was also pretty interesting.
Yeah.
So both of them kind of had the same kind of EME feeling thing where like we
update ZL, condition upon the input X and ZH, the last ZH, ZH T-1, let's say.
And then we keep updating ZL, ZL, ZL, ZL, ZL, and we keep updating it.
And then we go holding, we update ZH, condition upon ZL, and actually it's just ZL, it's not even X.
And then we just update ZH, and the way to think about ZL and ZH is ZL is like your
local scoped variables that are just being overwritten and updating, updating, updating.
And then ZH, and Alexia makes this point, that is a candidate
answer, a proposed latent answer that is just an embedding space away, a one MLP lookup away from the true answer.
So you're kind of like EM-ing just to like zoom out a little bit.
You're kind of maximizing the probability of the correct, you know, information stored in your
memory conditioned on a given output and maximizing the right output conditioned on
the information stored in your memory, quote unquote, in parallel.
And like that optimization algorithm leads to you ultimately learning a recursive method
that stores the right information to this local memory basically, and then outputs the right thing.
It really, like if we actually think of Sudoku, it's actually a really natural way to think
about what's actually happening under the hood, where Sudoku is an incomplete puzzle.
You can't guess every cell at any one time.
Actually, it's designed where you can only guess one or two cells based on the available information.
So it's an incompressible problem.
You actually can't do it unless you're just randomly guessing and guessing and guessing,
which is a very high combinatorial space.
And so what the ZL is doing is some type of, let me try this, try that, do some computation,
think about local things, and then it proposes.
And then we go to condition upon like something that it may have found.
It sends it to ZH, ZH fills it in.
And now we have a little bit more of a filled-in Sudoku puzzle.
And the training process is training the algorithm to know to do that, right?
It's like, it's maximizing that it's like, oh, this strategy for what you save
tends to lead to correct outputs.
Without chain of thought.
Without chain of thought, exactly.
That's the most important part, is that if we had Sudoku and we know how to solve Sudoku,
because like we were just, you know, dumb homo sapiens that didn't know how to solve Sudoku,
like it would just have solved it.
And that's why it's cool, because it actually is able to discover things without being teacher
forced via chain of thought.
Right.
Interesting, yeah.
Should we look at some code?
Let's do it.
Okay, let's dive in.
And I would love to see what these papers or models look like, just distilled down to their core essence.
I know there's lots of details on how you train them, but kind of the core training algorithm,
and it'd be great to contrast the two methods.
Yeah.
So, I mean, they're remarkably similar.
And so, largely one and learning one is learning with the other.
But basically, you start out with some ZH and ZL that are just zeros.
Yep.
You have some input embedding space to go from XRAW to X, which is the maze state or whatever
it is, initial maze state.
And then with no grad, you don't pass any gradients back through this.
So, this is the trick, basically, to not back-propped through time.
Here are two of the three recursion levels.
So, yeah, this is like the, they do this just for simplicity.
But I hit ZL, T low times, and then once for modulo, T low, then I hit the ZH, and I do it again and again.
And like you said, I'm updating ZL condition upon ZH and X.
Right.
And then I update ZH condition upon ZL.
And that's it.
So, this is the like expectation maximization style approach.
Exactly.
Approach, yeah.
And then you don't really need this.
This is like just for cleanliness to show clearly that there's no gradients occurring above this line.
Just freezing the weights past that.
Exactly.
And then I hit Lnet and Hnet one more time, and then-
Which is the same thing as up above.
So, this is just, okay, it's literally just the no grad thing running one more time.
Exactly.
Cool.
Yeah.
And just make it really clear.
And then there you go.
And that's your HRM model.
Cool.
That's quite simple.
Two and two is completely sufficient.
If you actually go much higher, Konstantin showed very clearly that it doesn't actually help.
So, that's two of the three recursions you said.
The third happens in the actual train loop.
And the third is in the train loop and at the test loop.
They both have this m-test or n-supervision, which Alexia calls deep supervision.
They call it adder refinement steps.
It's just whatever you want to call it.
Call it n-sup.
And so you do this n-sup times during training.
And then during test time, there's a different hyperparameter for how many times
it recurses over each model, which is m-test basically.
They're actually the same.
And so this and this, we can probably just call this the same.
Yeah.
But it's the same.
And if you actually, Konstantin does a good job of this.
If you actually train on 16 and you test on only one, you get like 7/8 of the performance.
Or like almost all the performance.
So it's actually quite interesting that this is just too much compute.
And it doesn't actually help you all that much.
So setting this to 1 is actually like pretty much--
But presumably for like more complicated problems, having more test time compute is still useful.
It's like the reason you would set it up this way.
Yeah, for sure.
And so we call our HRM.
We get some loss.
We backprop through just those two little parts here.
And then we step.
We zero out the gradient, but we do not update ZH and ZL.
These are still the same in it.
So that's the really important detail there.
Right.
And then so we go back.
We pass in the ZH and the ZL from the previous one.
So now this is actually not the same batch.
Right.
Because we have updated ZH and ZL.
So it's in a different part of the latent space.
Cool.
And that's the key like mini-batch construction through memory space concept.
Yeah, exactly.
And then at test time, it's simply the three loops.
So there's your outer refinement loop, which turns out like just at train time.
Mostly it doesn't matter.
Train time recursion was important, but test time recursion was actually not that important.
Yeah.
Which is kind of counter-intuitive.
And then the HRM inside that has your two other loops.
Makes sense.
And that's it.
So pretty simple.
Okay.
Now the TNR.
The only two changes, the main two changes here, is that they collapse Lnet and Hnet into just net.
Great.
And it's important detail.
These are four transformer layers.
This is four transformer layers.
And this is just one transformer layer.
And Alexi actually shows that going deeper actually didn't help.
Yeah.
And actually on some tasks, it was just the feedforward net actually worked just as well as a transformer there.
It was like on Sudoku I think.
Yeah, on Sudoku, MLP actually outperformed the tension.
It scored zero on the maze.
The MLP scored zero on the maze.
And so it's not clear, it's not obvious that the transformer is always better.
So there's the weight sharing.
And then instead of going back just the one, two, this back propping through just these two,
you actually back prop through one latent recursion step.
All the way through one latent recursion step.
So let me just walk through this a little bit.
So we have the same thing here.
Same starting point, yeah.
It's mainly the same thing here.
We're doing this six times.
And then we go one more time here.
And then we do our deep recursion.
This is the outer loop n sub times.
And so again, we have the no grad, we have the detach.
And then this is where it's different.
So I am calling this latent recursion after the detach.
So it's one full recursive loop is happening.
Versus here.
And so that's the main difference in the optimization.
Otherwise, it's effectively the same.
And then it outputs.
And then you're good to go.
And you train it exactly as the same way before.
And then at test time, it's the same thing again.
And so largely the same.
Cool.
And so in many ways, it's sort of a simplification, right?
You're collapsing certain parts of it.
You're simplifying this net architecture.
It's slightly more complicated along this back prop through time part.
Because you're actually back propping through more than you did before.
Right.
But it's like taking a bunch of lessons from the first one and basically simplifying most of it.
Right.
Which is actually, I think, is why she needs to make the model smaller.
And so it's a 28 million parameter model for HRM.
Now she brings it down to a 7 million parameter model.
It actually gets from 70% to 87% on ArtPrize 1 and does actually quite well on ArtPrize 2 as well.
And so, yeah.
So she makes the model three, four times smaller.
But because it has that recursion, it actually outperforms.
And there is this researcher named Melanie Mitchell that writes this book talking about this
very phenomenon which is like it is sufficient, not necessary to go bigger and get better performance.
And it is sufficient and not necessary to add more recursion.
And so where I'm really excited is what happens if you do both.
Right.
And you're still limited by back prop through time.
Even Alexia is limited by that last step from a memory perspective for sure.
And so if you can make the model really big and you have lots of recursion and we do something
else other than back prop through time, then we can get all the benefits of this and all the benefits
of the giant LLMs and then you can get some crazy stuff.
So now to wrap up, why don't we talk a little bit about the bigger picture.
What does this mean for the field of AI research?
How should people think about where these models fit into the current span of research happening,
especially given that it seems like a bit of a departure from a lot of the methods that people
are used to hearing about and increasingly seeing products that people use?
Well, I think for one, from the arguments that Schmidhuber makes and that we've talked about today,
recursion is important and it's not going away.
And clearly the benefit is here of adding recursion into models.
And you've seen things like the recursion language models out of Google that are pretty powerful and cool.
And so that's definitely one piece that's I don't think going away anytime soon.
The next one is this adder refinement loop, like tbtt, like t equals one, truncated back wrap through time t equals one.
I think that that is a really powerful idea and the fact that that works so well.
We have yet to really explore that extremely, really understand what's happening there.
And then the third is that idea of like, okay, we know that recursion works.
We have these tiny recursive models that are seven million parameters.
It can solve a hundred million, a hundred billion, a trillion parameter model can't solve,
trained on the entire internet and a seven million parameter wins.
Like the right answer is to like take the amazingness here and take the amazingness here,
which probably is already in Gemini already or some of these, it might be at least in some part.
But when you take the benefit of both these TRMs and these giant models and you actually slam them
together, I think that it's just going to take off and it's going to be really huge.
Yeah, one of the things that's really interesting about these TRMs and HRMs is they're not general
purpose models, right?
These were task-specific models, right?
The model trained to do Sudoku cannot do ArcPrize inherently.
It has to be trained on the ArcPrize set to do so.
Versus the LLMs that are used on these tasks are general purpose models that maybe get some
additional fine tuning data or in context learning data on those tasks.
And so I think that's where the interesting overlap might come is if you can make these more general
purpose agents that can somehow be general purpose in the way that the sort of next token prediction
algorithm has given us and do more complex reasoning to achieve that, it seems like you can
have really efficient architectures to do scale-up reasoning.
Right, a lot of the view of what these LLMs are doing is finding really amazing embedding
representation spaces.
Yes.
But reasoning inside that space is actually not done all that much.
Yeah, it's always through the token space.
It's always through the token space.
And so what you can imagine is we found mapping from token space or from vision, from pixels,
some really cool latent space where things are just nicely semantically separated.
And we can, you know, makes it really easy for downstream tasks to do.
But now in that space, use this, like tiny reasoning models, use some type of
recursion inside that and train those, that model on that, a little small model on that reasoning space.
I think that's really going to work.
Francois, thanks so much for breaking it all down for us.
See you all in the next episode of Decoded.
Thank you.