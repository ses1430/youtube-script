---
title: Building AlphaGo from scratch – Eric Jang
uploader: Dwarkesh Patel
channel: Dwarkesh Patel
channel_url: https://www.youtube.com/channel/UCXl4i9dYBrFOabk0xGmbkRA
duration: 9437
upload_date: 20260515
webpage_url: https://www.youtube.com/watch?v=X_ZVSPcZhtw
id: X_ZVSPcZhtw
categories:
  - Science & Technology
---

# Building AlphaGo from scratch – Eric Jang

today i'm here with eric jang who was most recently vice president of ai at 1x technologies
before that senior research scientist at what is now google deep mind robotics
and you've been on sabbatical for the last few months one of the things you've been doing is
rebuilding and improving and hacking on alpha go and so we're today we're going to do is you're
going to explain building alpha go from scratch and what it tells us about the future of ai
research and development but uh before we get to that why is alpha go interesting why is this
why is this the project you decided to do on sabbatical rather than just hanging out at the
beach sure yeah um i like making things and alpha go and go ai is one of those things that really got
me into the field when i saw the kind of early breakthroughs um on alpha go in 2014 2015 2016 and
so forth it was just profound to see you know how smart ai systems could become and the the kind of
computational complexity class that they could tackle with deep learning um this is a problem that
has you know long been understood to be kind of intractable for a search and yet um it was solved
um through through deep learning and so so that was quite mysterious to me and i've always wanted to
understand that phenomena a little bit better my training is often in deep neural nets for robotics
where it's uh the the decisions made by the neural networks are a bit more intuitive but alpha go is a sort of
problem where the the decisions are actually the result of a very very deep search and it's always
been very mysterious to me how like a 10-layer network can sort of amortize the simulation of something so
so uh so deep in the in the game tree yeah interesting so if you plot out how much compute it took to
build various iterations of strong go bots over the years you can see that in 2020 there was a open source
project called katago by david wu from jane street who who basically achieved a 40x reduction in compute
needed to train a really strong gobot tabular rasa i'm not certain if it's stronger than alpha go zero
or alpha zero or mu zero but it's very very strong and this is what most go practitioners today train
against when they're when they're playing an ai and thanks to lm coding what took a whole team of
research scientists that deep mind and you know millions of dollars of research and compute can now be done for
you know a few thousand dollars of rented compute okay i guess we should first discuss how go works
great so yeah how does the game work um so the game of go is a very simple one that can be implemented
quickly and easily in a computer the the objective of the game is basically to put down black and white
stones and try to occupy as much territory in the game as possible so i might start by putting down a
black stone black always goes first so go ahead and so the way you capture an opponent's stones is that for
every um intersection if you can surround all four of its neighbors with um with your stones then um then
this one is sort of cut off from oxygen if you will and then it uh it is a dead dead stone so so then
now i control these four stones as well as this empty intersection here so there's like slight variations
between chinese japanese and what is called trump taylor rules trump taylor rules are designed to be
completely unambiguous for go so this is what all go ais train against and resolve against so in typical go
like when the humans play you're actually not allowed to put this white stone down here it would
be instant suicide um in trump taylor it's actually fine you put it down and then it immediately resolves
to death so the outcome is sort of the same let's go ahead and start over and play play a few stones
and then i'll explain some more so i'll just start there all right i'm like basically playing randomly
here but i'm trying to get around your stones and see if i can close one of them
so this move um basically exposes one empty neighbor for your white stone and it's very akin to
a check in chess where if you don't respond immediately by putting anyone here then i can
immediately capture this i see okay because it is it is sort of the diagonals that determine whether
you're grounded in the cross-section not the diagonals so so this one is surrounded on three sides yeah
and so um you're at threat of losing that stone if you don't play one immediately there now you can see
that i'm starting to pressure you because by putting a stone uh here now you are forced to um put one
here otherwise you would have this two block yes to yourself and then if you think think through like
what happens if you were to respond here you can probably you know search into the future and deduce
what i'll do in response uh once you once you do that you have a lot of confidence in my abilities but
i'm guessing you put the black here that's right and then i would capture all three of these stones so i
just assume that this is gone this little block is gone yes so in go it's actually okay to let
opponent capture um some stones if for example it allows you to position uh to capture more stones
in somewhere else on the board and this is what makes go a very beautiful game is that um you can
kind of uh lose the battle but win the war right and and as the board size increases the complexity of
these kind of like micro versus macro dynamics uh gets gets more interesting but presumably you'd put one here
and so now i would capture this entire group okay and this would be mine okay there's one more um case
that i want to demonstrate which uh actually i had a bug in my code uh recently which is the following
situation so let's consider a formation like this right and then you know we have other pieces on the
board in play or whatever um and so um let's talk a little bit about how the game ends right um in this
territory who controls these these areas is it white or is it black white it's actually black because i
have actually surrounded this whole area yeah and it's um very assuming i have like other black stones
here it's actually very hard for you to break this out of the control of these so when the final score
is tallied would would these ones also count as being in um yeah great question so so um this is where
different rule sets have different ways of scoring and so we should talk a little bit about how like
you resolve scores between humans and how you resolve scores between computer code because there's
actually some ambiguity in how humans evaluate this so most humans would look at this board configuration
and conclude that like black has kind of totally surrounded white and so white has no chance of life we
could play out more here but then at the end i would capture everything yeah however if you have a way of
breaking this formation and connecting white to something outside of it then it can flip right and
so this is where it's you know a little bit hard for a computer to decide these kind of things right so
how do humans do it right like it's worth thinking a little bit about how humans resolve this because
this will actually map later to how we think about the deep neural network um humans basically say
uh i think the game is done and then you you have to also say i think the game is done and then we'll
say like i think these are these are my stones and then you have to agree if you don't agree then we
keep playing yeah so um essentially once two humans their uh so-called value function um agree on a
consensus then the the chinese rules uh result that yeah interesting so in trump taylor scoring um it's
perfectly unambiguous so it can be decided you know algorithmically by a computer so if let's say you
you have this at the end game the way you score this is that you first count how many stones you
control and that's unambiguous then you count how many empty intersections that are not touched by your
opponent's stones so these intersections would not count for either player because both of all of
these intersections are connected to both white stones and black stones right if um this were like this
yeah then white would get three points now this is a little odd because a human would know that white
is actually losing these points yeah but trump taylor's scoring would consider white to have
all these points as well as these points you got it okay so so that is a very big difference
in um how computer go scores things and how humans score things how does the game end the game ends when
either a player chooses to resign or both players pass consecutively cool yep so that's the rules nice all right
now help me correct this with ai great okay let's understand how um alpha go actually works and how
somebody in the audience might be able to implement it great yeah let's start with um kind of an intuition
about the underlying um you know search process used to make moves and we'll layer on uh ideas from deep
learning to make it much more efficient and attractable so go is a game where there's just two players
we're going to draw a person here and we're going to draw an ai here and um let's say this person is
playing black so they go first so we're gonna draw
here and then now the ai is going to make a move based on what it sees here so there's a question of
like how you encode these inputs into the ai maybe you could use ones and zeros but you want to represent
um you know black white and empty so so you would need at least three different values here right so maybe
you could use zero ones and twos or something so so the ai might see something like you know zero
zero zero zero zero one
great so so this is the input to the ai uh on its turn yeah so so the ai can choose let's just pick three
possible random moves that i can go and i just drew these at random and so which which move is best
here right well we don't know until the game ends um there's no go does not have any kind of
local reward of which move here is good and this is what makes go a very difficult game is that you
don't actually know who won until you really get to the end of the game so how deep is this tree
right well in a 19 by 19 um go board there are uh you know roughly to the order of 361 moves on
any given uh move and of course as it fills up you have less moves um and and the the number of
steps in the game can be somewhere from 250 to 300 moves and maybe experts might uh decide to end the
game uh well before that but uh you know under trump taylor scoring you actually have to play things
all the way to the end so this could be like 300 moves or something right so like 300 um like depth
of the tree yeah so if you keep on expanding possible moves here so in in this move the ai is going and
then you know here the human would go and then you know there's there's some
and so forth you can find that like essentially what you end up with is an enormous explosion in the
possible game outcomes originating from just this one state so this is something to the order of like
you know 361 to 300 power of 300 which is far more than the number of atoms in the universe right
like it's it's just uh it's just and of course actually there are redundancies and symmetries so
it's not actually 300 but but that's sort of the if you were to do a naive tree where there were no
merging of children then actually you end up with a tree about this big what do you mean by merging of
children right let me uh use this board here so if we start here and then you play here and then
i play here and then you play here that is equivalent to i start here you play here i play here yeah and
then you play here right yeah so so both of them arrived at the same spot but through different paths
so this child node can be thought about as a shared ancestor yeah and i guess it's not 361 it starts at 361
but it decreases by one each time and the branching factor decreases by one each time yeah yes yes
but in any case this is a very very very large tree and this is also why you know computer scientists
for many years thought that go was not a tractable problem this century because the amount of compute
you would need to exhaustively search every possible possibility is just too large if you could go is
actually deterministic game so on any given state you can actually compute what the best possible
strategy you can you can make is in order to win the game you can search all the possible futures where
you win and then just make sure you always stay in in that you know set of futures um so alpha go's kind
of core conceptual breakthrough was using neural nets to make this search problem tractable so before we get
into you know how neural networks are involved let's talk a little bit about how we can you know assuming
we have a powerful enough computer um search this uh this tree to find the best move right so in the
beginning um you're not going to build out the whole tree uh because storing that tree would be very
expensive instead you might do something like interactively figure out which um which leaves of this tree are
worthy of exploring and expanding into the future to see you know what else is there so um there are some
early algorithms in uh bandit literature like you know ucb uh one which is not exactly appropriate for a
you know sequential game like go but very much inspired the action selection um algorithm used in uh in alpha go
so so ucb one looks like on on every move we're going to take the best action or you know the arg max over a
that maximizes um you know the q of a and i'll explain what q of a is in a moment plus some sort of exploration bonus
so on every node we're going to track a few quantities so so let's you know consider each of these a node this is this is the the root node
um where you're making decisions from and these are the children of the root node and um we're going to
say each node is basically a data structure that is um it stores a visit count of this um this node this
this child node is how often the parent visited this node yes and we'll call this an action so so one thing
that is easy to trip on and is like if you come from you know robotics or um other kinds of reinforcement
learning is like where are the actions right i'm only talking about nodes um nodes here represent states
and because this is a perfectly deterministic game with no randomness you can actually just infer the
action based on the child so so if i go here that implies an action and this is the state that we resolve
right so so the llms if you ask to uh you know vibe code a mcts implementation it'll most likely design
the right data structure here but um you know it's up it's sort of a chef's choice you can actually
rewrite the tree structure however you like this was what um claude 4.6 wrote for me when i when i asked it
and it was a very reasonable choice so um so then you know q represents the um mean action
value of this action and i'll use a subscript a to denote that this kind of corresponds to taking a
specific action to to get here right from from the from the root node um so so like uh if we have root
basically taking a gets us to this this node here um and then we're going to also store the probability
of taking this action again from the parent from the parent yes like like what are the odds that we
sampled this one yeah and and this will become relevant later you know like uh we've talked about
a deterministic tree for now so i'll bring probabilities into this later and then finally we have a sort of
dictionary of children which is just like you know more of these notes you know in a sort of classic
linked list style reference tree so um this is the basic data structure to implement a tree
and um in alpha go they use a slightly different action selection criteria called um pucked and it's
short for predicted upper confidence with trees and uh this is basically um when you when you select which
which which child to take you do argmax a of q of s a plus a constant
so the equation and forms are actually pretty similar um these are both scoring
criteria right like you want to argmax this quantity and you want to argmax this quantity to
determine which action to take so let's break down the intuition of like how you select actions here
this is the mean action value so how good is a given child on average um and and and if you
actually you know knew the whole tree then uh this is all you need right to select the best action
you don't really need to do more than that but if you're interactively building this tree as you're
figuring out what the q values should be then what you have to have to do is occasionally try some other
actions you know as a sort of explorer versus exploit trade-off so in both ucb and pucked there is this
term here that basically rewards taking actions that you haven't taken before so as we mentioned before
each node stores the visit count of taking that specific action right so everything is initialized to
zero and so for a given action let's just say like call it and like action a initially it's zero and and so as n is
increasing if if let's say we've already made 10 um 10 action selections from that root node uh but we
haven't picked a yet then this term actually starts to become quite large for a yeah right and conversely
if we have chosen a 10 times out of 10 then now this term is quite small it diminishes very quickly
and the same thing is actually true here just make sure i'm understanding it maybe i can
put it in my own words let's just focus on ucb what we're saying here
you can think of it conceptually it's two different things
the q and then this exploration term let's just be clear about what q is q is basically saying hey
once we do these rollouts so you're actually running all these simulations you go down the tree
and then you figure out okay if i end up at the terminal value of this tree do i win this game or
not and then you do this you average whether i win this game or not across all the you know the
leafs of this tree from starting from this node that average you put in q correct and so you're saying
the q is basically representing will i win this game or not uh what is probability they'll win this game
starting at this node that's your sort of um that is your sort of exploit that is like saying
i've run these simulations i think this is a good move or not and then this other term is saying
um have i explored this branch enough yet relative to the other actions i could be exploring
or i have already explored uh if i haven't explored this branch yet you know maybe i think
it has a low score but i just haven't explored that many branch leaves of this uh down this
uh leaves down this uh down this node in this tree so i should maybe like try this even though the q
that sort of exploit is telling me that this is not that valuable and because ln of n grows slower
than n uh basically as over time you will move from the argmax being dominated by this exploration term
which is the second term here to the argmax being dominated by the q term which is like okay i've done
enough simulations i'm quite quite confident that like this is the branch to go down yes that's right
so um the motivation for ucb was to come up with an algorithm where if you don't know the payoff of
the arms uh the different actions you can select to begin with this uh strategy basically with given
some exploration term here bounds your regret uh in terms of how wrong you can possibly be yeah um i don't
know the proof i don't also know if this one is proved to have a logarithmically or or like uh you
know square root bounded regret or anything but i think the algorithm was just derived to look
something like this and you can tell that these terms are they grow a little bit differently and
this is actually just to account for the fact that go has many more actions in every given move
compared to your standard banded problem yep um so uh one small clarification to make is that you
talked a little about simulations on the probabilities and forth um we should remember that go
fundamentally is a deterministic game so the notion of like where does the notion of probability come from
here right um if you had a very powerful computer there is no probabilities you just you can just compute the
true average of what the the mean action value is so where does the probability come in well it turns out
that um uh as in you know computer go before alpha go we've always done some sort of monte carlo method
where we have some we we take the expected q value averaged over a randomly selected tree um and that
randomly selected tree is where probabilities come in so the interpretation of q is um what is the expected
action value under the um under the random distribution induced by some random search process makes sense
and so where does the random search process come in that's where uh you know p of action comes in yeah
so if we assume a very naive algorithm where you have a uniform probability of taking any valid action
then this would just be one over you know the number of valid moves in uh in the in this setup and you
would be kind of taking this average over this very diffuse tree right and and this is uh this is a valid
integral you can take but it's very slow because you're going to consider a lot of trees that have
very low value and uh it's essentially almost like a important sampling problem where you want to
there's only a few actions and and sort of uh paths that can contribute you know high value and almost
everything else is low value so so that's a sort of a tricky um problem here okay um so this is the action
selection criteria for how you decide which moves to move down now as you move down um in in tree search
you will eventually run into a node where um it's quite clear you've won or lost right at the at the very
very end of the game when when there are no valid moves to play left under under trump taylor's scoring
you can decide whether you like you know won or lost right so you either win or you lost and so this is
basically um you know the the final return of the whole game right um and so the the question here
is like we we can assign a um a value u to a terminal leaf node of the tree but how do we assign the
values for nodes prior to that the parents and it turns out um you know what you simply do is you just
take the um your mean action value is essentially your average so let's suppose these were leaf nodes um
sorry these were all leaf nodes the the mean action value of this node you know this action here
is just the average of whether you want or lost at the leaf nodes and uh correspondingly you can kind
of walk up the chain and say like well the mean action value of this node let's call this like qb and
this is action b is just the average of a weighted average of these ones here yeah right and and the
weighted average is um it could be dependent on if you have a different sampling distribution or not
but the that the basic intuition is that you want to resolve the game where you have a deterministic
win or lose and then you can kind of go backwards uh this is called the backup step and assign values to
these uh these these these nodes or actions um uh corresponding to the averaged over over the final
terminal leaf okay so um if you were to do this without neural networks it would still be intractable
um you would you would have trouble finding you know which um actions to sample a lot of the actions
will contribute very low value especially if you're like you know trying to fight your way out of a
losing position and only a few actions give you high value so the search in practice is still very
very expensive um but but the the idea is that like if you can because go follows a tree structure
you can actually you know inform a very good estimate of the value of this node based on the uh values of
downstream assuming they're all correct and assuming you've searched deep enough your explanation
earlier about the um the sorts of states where it's obvious to a human who's going to win but it's
not obvious to or like you deterministically you still had to play it out actually drove home
the intuition of why the value function both is trainable and to why it's necessary in order to
actually be able to learn this game effectively i mean each word defining value in the first place but
sounds good yeah yeah so we we talked about uh you know this you value being you know your final resolution
of whether you want or lost and this is the terminal leaf node condition um now humans don't play all the
way to the sort of edges of the the tree the leaves of the tree right they kind of stop you know some
dozens of moves before maybe maybe even 100 moves before in in sort of high level play so how do they
know right like you can think about humans as implicitly having a neural network called a value function
that basically um you know takes in a board state and then kind of evaluates um you know
he went and so the human glances at the board and they know like i'm probably going to lose right and
they're essentially running a neural network that looks at a board and implicitly they are amortizing a
huge number of possible game playouts and and taking that average and then deciding whether the board is
winnable or not and then whether they should concede or or you know keep playing or not and uh this is
remarkable if you think about like the um the beauty of something like this it's like a neural network
in a in a human can somehow do all of this simulation at a glance and then just know like within a few
seconds without actually playing every single game logically based on just kind of like crystallized
knowledge and experience that like they can do this and so this gives us a hint that like in games like
go um there are ways to basically radically speed up the search process and this is one of the fundamental
intuitions behind why alpha go works is that you can train a value function to look at a board and
quickly resolve the game without playing out all of these trees into the you know into a very deep search
depth yep makes sense i will say for the audience um i sort of found uh for previous episodes when i was
prepping and it would seem somewhat relevant to understand how alpha go works i would find it very
very confusing and but it's the kind of thing where once you understand the problem in this way and then
you'll build the next few pieces it is actually much more understandable and it will make a lot of sense
and it's okay to be confused right now uh but it's it's probably simpler to understand
by the end of this lecture than you anticipate so i'll just make that note for the audience uh yeah
the important intuition at a high level just to you know step back about where we're going with all
this is that um classically for games like go you could build a tree but we don't have computers
powerful enough for that yeah and um estimating the value of every action that you could possibly take
is also hard because you don't know until the end of the game you could take averages uh by playing
them to the end but that's also hard because you don't know which actions to take to sample these
averages so conceptually there's kind of two problems there's the breadth of the tree and then
there's the depth of the tree and alpha go gives us a way to basically uh shrink both of those to be
very attractive yeah that's that's essentially the kind of core idea behind it okay so we uh we take this
idea that like you know humans can glance at a board and instantly predict whether we win and maybe that
gives us the opportunity to really truncate the how deep we search yeah and then you know we also know
that humans can look at a board and um and uh decide you know um what what boards you know
like intuitively at a glance what moves might be good on a good board right so so these are kind of two
things that we can use deep neural networks for to accelerate this search process um let's go back
before we talked about neural nets let's just go back to how this play out works so we've only talked
about making one move right so so the ai looks at this encoded go board it has a tree um it searches
for you know deeply into the tree to find out which of its actions might be the best and then it takes
that action and then now you know it goes back to the human so maybe now the human sees uh a go board that
looks like you know like this and um and then they um they make their move so maybe they put um they put
their stone here and then now we um we go back to the ai which now looks at a new encoded board
so i've used two to denote the ai's playing as white and one to denote the human playing as black and zero as
empty and then now on the ai's turn it does the mcts tree search all over again from scratch right so
so it throws away this old tree that is searched last round and now there's a new root node and it
begins to search a new and then so and so forth so mcts is basically a you can think about it like
a search algorithm that is um deciding what moves to play best aided by neural networks um and and it's
it's done on every every move okay great so let's talk about the neural network part of this and while
you're racing another sort of thing that was important for me to understand was this mcts data
structure with nodes and children's of nodes and whatever um this is done per move and reinstantiated
once a move is made so a human makes a move then the ai looks at this and is trying to basically
run a bunch of simulations uh to figure out okay what should move should i make next and those
simulations just a simulation is basically like exploring one more node in this mcts tree and at the
end um once all these once all this you know you run a thousand simulations that informs then this um
i guess as you'll explain this probability of what move to make next that's what you store you sort of
choose the best move given those probabilities you discard all of that then the next player makes a
move and you restart this process at the beginning of every move correct one small addendum you don't
discard all of that you keep one thing behind that we'll use later yeah just like i did for reiner i
wanted to make flash cards for this episode so that people could retain these concepts and ideally an
llm could generate some candidates for me to then refine but to actually get high quality suggestions i
needed to design a whole pipeline where the ai could take and ingest screenshots of the blackboard and the
right time stamps and then make svg diagrams in case visuals were helpful and then run the writing and
drawing through a critic and then revise the card in response to this feedback it's very hard to
accomplish this just by stacking llm calls this sort of step-by-step recipe works much better if you have
a durable agent that's been engaging with the task across all the previous stages so i used the cursor
sdk to spin up an agent for each card the cursor hardness saved me a bunch of work in designing some
custom context scaffold or figuring out how to design tool calls for taking screenshots or making
animations these agents all run in the cloud so i don't have to worry about leaving my laptop open
i just get an email when i have candidates to review you can check out my cards at flashcards.thorkesh.com
you can start building with the agent's sdk at cursor.com/thorkesh okay so now we have a basic
intuition of how moves are made with search we're going to talk about how neural networks can speed
this up by providing an analog to like the human intuition so there's two networks there is the value
network which takes in a state and it predicts you know am i going to win or lose it's a binary
classification problem then we're going to have a policy network which induces a distribution over
good actions to take so i'm going to draw a one-dimensional flattened move distribution
but this is really like you know a square kind of grid right so so maybe like it thinks actions are
like these are the kind of probability distribution over good actions and both of these are categorical
classification problems right so you can train this like any classifier in with deep learning
you know cross entropy loss that kind of stuff so the um the specific architecture does not actually
matter too much i tried a few different architectures transformers work resnets work for small data
regimes uh my experience is that resnets still kind of outperform transformers and um and kind of give
you more bang for the buck at lower budgets but this may not be true why is that um they provide the
inductive bias of like local convolutions and generally transformers start to outperform uh residual
convolutional networks when you want more global context i see so so one um interesting finding from
the katago paper was that they found it actually quite useful to pull together global features together
um and aggregate global features like uh throughout the network um to kind of give the network a global
sense of how to like connect value from one side of the board to another side of the board what does it
mean to aggregate global features yeah so if you have a um go a very large 19 by 19 go board yeah and
you you know you've got some some sort of battles going on here and you've got some battles going on
here um when you pass this through a convolutional neural network the receptive fields of the
convolutional network are going to be good at computing local things and making that invariant but um
they won't be able to kind of connect these two features easily right they need to sort of be
pooled together and attend to each other somehow so the argument about you know why transformers are
good for computer vision tasks like with uh you know vision transformers and so forth is that
because they have a sort of global attention across the whole thing they can more easily draw
these connections but you do need more data there so that you can kind of uh learn through data the
the sort of invariant local local features um i've tried very hard to make transformers work for this
problem because i was kind of curious if transformers would present some sort of breakthrough and go and
just remove a lot of those tricks but to try as i might i actually haven't figured out a way to
make transformers better than resonance for now so one uh sorry one more potential question
it makes sense why transformers with their like global pooling of information would be better if
you need to consider information that is not just spatially um uh or yeah cnn's give you a sort of
bias that the things that are next to you are especially irrelevant and then they're sort of aggregated up
yeah exactly yes but suppose okay so for games where it isn't that relevant what is happening locally
you just kind of have to consider the whole thing you're saying transformers would work better how about
games where so they're talking about the spatial dimension how about the temporal dimension where
right now we're only considering the previous move because it is a deterministic full information game
where um but what if it was something like poker or diplomacy where really a bluff they made a while
back is sort of relevant to understanding now and isolating to decide to make your next move and so
you need to consider all those previous states would that then change the consideration of what inductive
bias is most relevant and what architecture is most relevant right great question so go is a
perfect information game yeah and in perfect information games um there does exist a nash equilibrium
strategy for which you can do no worse than any other strategy so um if you know that your opponent has a
particular bias like they they love to play aggressively you can actually in principle counter that specific
strategy better than a national equilibrium policy but um to counter any given strategy um there does exist a
single um national equilibrium that can be decided solely using the current state so um that that is a design
choice that most go agents alpha go chose to do which in hindsight turned out to work very well because
the uh nash equilibrium seems to be superhuman like like no human strategy seems to be able to beat it
now there are variations of this where you would actually need to consider temporal history
so and and this is a very exciting research area that i would encourage people to kind of
fork my repo and try these things out which is um if you were to play let's say 2v2 go then you actually need to model
your partner's uh behavior and you like you may not have information on how they play so you need to
aggregate some information on like how they play so that you can respond accordingly yeah right
like these are uh situations where it's no longer a perfect information game yeah and then in those
cases in in games of imperfect information or partial observability then you do need some context to
build a model yeah and and i think that's a place where things can get very very exciting in terms
of like self-play or you know diplomacy style yeah interesting okay so uh returning back to
the neural network the architecture again is not super important you can get it to work with transformers
you can get it to work with resnets i found that for low budget experiments uh resnets work a little
better um you can also use kind of a karpathy style auto research hyperparameter tuning to make make your
architecture pretty good and so so you don't have to worry too much about that you just need to sort of
set up the problem so that you have a sort of target optimization yeah okay so we're gonna pick just a
somewhat arbitrary architecture that worked for for you know what i did but again this part is not super
important um you have your encoded board state and uh we're gonna just choose to let's say do three uh
three like you know similar to an rgb we're gonna have three kind of channels uh one channel to include black
one channel to include white and then um and then uh one channel maybe to encode like um uh empties or
um maybe like a masked region if you want to train on multiple board sizes i'm actually not going to
talk about multiple board sizes for now that's a little bit too complicated so we'll just say like
you know we've got this two or three channel uh rgb like image and then we go into a you know a resnet
and then we have two branching heads um one head predicts the the value function and this is like a
a single logit so this is like r1 and then we have the policy which is you know r361
so this is the architecture and uh we're going to basically train this to predict the outcomes of games
given the board state and we're also going to train this to predict what are good moves yeah right so
the og alpha go paper uh or called alpha go lee um initialized this network with a supervised learning
data set of expert human play later they remove this restriction by having the model teach itself how
to play well but i find it actually from a matter of like implementation for your audience super super nice
to always kind of initialize your your experiments to something that's easy and then like you know get
the problem working before you know trying to bite off the whole thing and learn a tabular resin you
you generally want to kind of initialize just as in deep learning in this initialization is everything
right um you always want to initialize your research project to something as close to success as possible
especially if you're you know doing something new that you haven't done before like always pick
something that works and then get it to do something better rather than start from something that doesn't work at all and then you know
make it work so um under that philosophy it's a great idea to start from something that like you know has a good
initialization so we're going to take human expert plays um and train this model to predict um you know good
actions right so we're going to take all of the winning games all the moves in which a human won
and um sorry an expert won and then predict those actions and then uh regardless of board state
like you know whether you want or lost you're going to predict the outcome yeah so you might be
wondering like okay well some of the early boards you know where basically only one stone has been put
down how could you possibly know whether who the winner of this game is right well if you have uh you
know hundreds of thousands of games then in on average you'll probably see that boards that start like
this have a sort of half of the games that branch off from this will win and half of the games that
branch branch off of this will lose so that'll actually be fine when you train this model to predict
those the logit will sort of converge to uh you know 0.5 um and and so so for these for these things it's
sort of expected that once you train the model a starting board state will look like 0.5 and then
as you progress towards the end of the game it'll actually look something like you know if this is
0.5 the the win probability will sort of either go like this or it'll it'll go like this right and
and this is sort of your move number yeah and so as you you know get hundreds of steps into the game
it becomes much more clear like who's more likely to win or who's more likely to lose under your expert
data distribution i i didn't understand the significance of why the this way of thinking
about values especially relevant to the expert data it is not relevant to the expert data it's true for
any data that you train it on yeah so if you were to learn a tablet rasa you would also expect this to
follow yeah so um if you just do this like so imagine you know you're vibe coding alpha go and you um
you you gather some expert data sets from like how to go online um or you you know you have a data set
of human players and you train this model actually it turns out this model is already a pretty good go
player it'll most likely beat most human players right so so like if you just take this policy
recommendation and take the argmax over you know it's uh if this is the you know probabilities if you take
the argmax and you just take this action as your go play um it'll be a very very fast go player that
doesn't think in terms of like reasoning steps it just kind of shoots from the hip and it'll be a very
strong go player which is already quite miraculous if you think about like you know 10 neural network layers
maybe under like 3 million parameters can already do something that impressive yeah um um and so you
can start this way and it's important when implementing this to kind of just verify that this is probably
true it's good to verify that your go rules are implemented correctly that like you know you can run these
simulations relatively quickly uh and and just as almost like a sort of a a checkpoint that like you want
to make sure that you can actually do this basic step before you try to layer on more complex uh things
like search yeah um so but yeah we can do a lot better than taking the raw neural network and playing
the moves and this is how we can apply it to monte carlo tree search so let's apply the neural network to
um to improve monte carlo tree search so we start with our root node
and we now have a four-step uh iterative process to do mcts so this tripped me up when i was first
reading the paper and trying to understand it but um uh essentially what we're going to do is we're
going to choose a number of simulations so like you know num simulations and this number varies this
can be you know somewhere between 200 to uh 2048 i believe in um in the alpha go lee match they use
tens of thousands of simulations per move because they really wanted to boost the strength of the model
as much as possible yeah um but in training you don't actually need too many and katago i think uses
something on this order as well do you know if they used uh if you watch a documentary they had a laptop
out during the game yeah they didn't use a laptop itself it was like on some it was on some tpu pod i
think yeah um but now honestly kind of unfair well uh like lee is not using like one e22 flops to
do a move you know fair enough um interestingly enough modern go bots don't need that much compute
at test time yeah and what we'll actually find out as we talk about how the mcts policy improvement
works is that over time the raw network actually takes all of the burden of that big tpu pod and
just push pushes it into the network and and you can do actually all of that work with one you know
neural network forward pass um but but the tpu pod will always add the extra oomph on top and so
that's what they wanted for the match so so we're going to pick this kind of like num simulations thing and
for every simulation we're going to basically do several things simultaneously we're going to see which which
moves are the best in the current tree we're going to add extra leaves to the tree if we get to a point
where we need to add a leaf and we're going to update the action values for for the tree so that's that's
what every every simulation involves these kind of like four-step process so so the four-step process is
is basically selection expansion
evaluation and backup
so so at the beginning of our monte carlo tree search our tree is very basic it only has the
the root node our current board that our ai wants to play out and so
we're going to basically select the best action for this so when this root node is created we're also
we also know that we can evaluate this under our neural network and get the quantities
you know v theta as well as our probability over actions
and i'm going to say root so for all of the actions here we can create a bunch of children right
so so this one has well in this case i'm drawing a three by three board with one one we're missing so
so basically there are um you know eight possible children uh associated with this root node
so like
and each of these has an associated probability of taking that action right so so there's p8 p1 p2
et cetera okay so at the beginning of our monte carlo tree search we have our root node and we can
initialize it with some children right because we know it's uh the policy network evaluated on the
root node gives us on a three by three board with one existing stone placed eight possible children
that this uh ai could take um so with each of the children their policy network also gives us the
probability of selecting that child so the um first step is to do the selection of the tree and again
this is a very shallow tree all we have so far is a tree of depth one essentially right so our first
move is to select by maximizing or arg maxing the pucked criteria which is basically you know q q s a
plus um c pucked times p of a divided by n over one plus n a
so for each of these we're going to uh you know n a is zero for for all the actions initially n is zero
and um and so we're going to basically just you know pick according to this um
initially uh what is going to be the you know chosen action here is most likely going to be biased
towards um you know the highest likelihood action here right because these are sort of uniform for
everybody so um let's suppose p1 was the highest probability node so you you selected this one here
now you got to this node and you realize that it's not a leaf node right there are more it's not a
terminal game so you cannot resolve the the final resolution so the next step that you do is um expansion
so you um you will then run this node this board state through the policy network note that this is the ai's
move right like ai is making this move and so when we expand this tree we're now thinking about what the
human might do or any opponent might do right so so this is like you know your your opponent um
the tree expansion process actually is completely uh so so when we evaluate the um the node here
we're going to now evaluate the the node from the perspective of this player yeah um so then this one
one has possible actions that we could take and uh we we expand basically the the leaf nodes here
so for each of these nodes um that we could you know arrive at we're going to now check how good those
nodes are right so so maybe um from here like the human could play here the human could play here or
human could play here and we're going to um store essentially the v theta for each of these things so v theta
of you know node one or like node one um prime v theta node one
um and and and so we're basically using our neural network to make an intuitive guess of how good is
this um board from the perspective of this player and uh fortunately because the uh it's a zero-sum game
it's easy to deduce that you know the value for this player at this this step is just one minus the value
for you know from this perspective so it's easy to flip the search process depending on which player you're at
um and so so this is the expansion step you've taken a a non-leaf node and expanded it and evaluated the
value and this is essentially a quick guess as to like if i were to play to the end am i going to win
or not right so you can almost think about the v theta as a shortcut for uh searching to the end of
the tree for for any given simulation um and then we're and this is this is essentially the evaluation
step we're evaluating the quality of each of these boards in original alpha golee they actually did
something uh kind of interesting which is that they took this value and they averaged it with the value of
a real go play out so they actually played a real game from here all the way to the end so like i'm
just going to draw this squiggly line to indicate some path and uh they kind of like play this all
the way to trump taylor resolution
of a full board and so this is like a zero or one right
and so they took this value and they just averaged it with with this one here so the the the formula they
did was like uh you know alpha times v theta of of like you know some some node um plus uh sort of
like one minus alpha of a of a true randomly sampled plant
and you might be wondering like okay well how do they play this out right like it would be very very
costly to do another search on on this play out like almost like a tree within a tree so they don't do
this instead they just uh take the policy network and play it against itself so they just take this as
both players and they just play it all the way to that and and um this is something that helps ground the
um the estimates here in in reality because you can get a single sample estimate of like whether you
win or not you can think about in the end game where the board is almost resolved that this one actually
becomes quite useful because the random the the play according to the policy will most likely
decide a pretty reasonable guess of the game and so you're not you know facing a problem where
this one kind of becomes untethered from reality it turns out this is totally unnecessary so in
all subsequent papers after alpha golee they just got rid of this yeah and so in my
implementation i also did the same and it speeds things up a lot because you don't have to
roll these games out on every single simulation yeah okay so uh again just to
reinforce my own understanding and just to re-explain it for the audience by the way in case it's not
obvious the p there in the select that is the probability coming from the network in this case
correct the policy network here yeah okay so fundamentally
a simulation just think of it as like rolling out one more node in the search process
um almost so a simulation is easy to think about when the whole tree already exists right you just
walk down the tree um using the puck selection criteria and you you uh and then and then you keep
going yeah now uh in alpha go the the data structure is such that we begin with a tree that has no
like basically only depth one which is its only children and you want to iteratively build out the tree as
you're also selecting actions down the tree so that's the kind of core thing here is that because go is
such a combinatorially complex game you cannot afford to build the tree in advance and then search it you
must search while building the tree right okay so um let me just finish up with actually the last step
which is the backup right so once you've scored these things you basically take the mean the the value
the q value assigned to the node here for taking this action is now just the average across your evaluated
values here it's you take a running mean over over uh all of the um the the simulations that you've
taken and they average the values of the children so that's what is known as the backup step and once you
evaluate this you can actually kind of recursively go back so if you know the you know the action value of
this node you can then take the average on its parent and so and so forth so so you have this kind
of four-step process where you are choosing the best action that you know of so far then you may
run into a node where you uh you you haven't been to before so you need to grow the tree a bit and then
you run it through the network to guess whether you're going to win or not and then you walk all
the way back up to the to the root node to update your values on what the best moves are so as you
do this iteratively this selection criteria will cause you to visit the because you're always selecting
according to this criteria you're always going to be selecting the best action you think at any given
branch right so so the final um visit counts of like how often you chose these things will reflect your
correct policy distribution as induced through this search process um and so so the visit count that you
store in the node earlier actually becomes the sort of vote for like which way we should finally select
an action here yeah so um you know as a sort of test of understanding it's worth thinking a little bit
about whether we could make this even simpler right like could we actually maybe even get rid of this
one and still make the thing work um so recall that you know when you do an expansion and then an evaluation
at let's say this node you you are checking the sort of win probability of each of the child nodes right
and so if this one is you know like one and these are zero um you do kind of know something about
which action might be better to take and so why would you need still need this right like why not just
normalize this one into some distribution and call that your your policy distribution um this is fine
you can do this and um this probably does work but in practice having a single forward pass that gives
you a pretty good guess is um is how the uh the breath is is is pruned out um the there is a sort of
duality here like it would be weird if let's say the policy recommended an action that disagreed with the
value right if let's say the policy said this was very high probability but this one said it was a
you know low value then there's actually something kind of fundamentally wrong between uh between
your policy head and your value head so they are linked and uh you probably could get rid of this
if you came up with a different way to recover this from just the value evaluations right but just make
sure i understand the reason you don't do that is so that you don't have to do 360 independent
uh forward passes to like hey here's the value of everything let's target max over it right instead you can just do one
forward pass and get like the probabilities of all of them um you can usually batch these somewhat
efficiently um so it probably is not a huge computational burden in practice but yes you
would have to pass 361 like up to 361 boards into a single mini batch update to evaluate all the values
here then normalize them um now there's actually a more uh important reason why we still do this which is
how monte carlo tree search is used to feed back on itself um and and sort of recursively improve its
own predictions and search capabilities and that's where this this one having this as an explicit entity
you're modeling rather than an implicit normalist normalization over your value is is a good idea makes
sense okay okay so um so we talked about the simulations and basically you know what you end up with
as you roll out the number of simulations is a tree that kind of looks like
i'm drawing a very low dimensional version of this of course it's in in in the real game it's much more
high dimensional but like you'll end up with basically a tree structure that like has um
a lot of leaves that kind of terminate and are not visited again because their value is deemed to be too low
but then you know along one path there will be a set of actions with very very high visit counts that
kind of gravitate towards that one set of decisions as you increase n so this is kind of like the mental
picture of what the tree in monte carlo tree search looks like and you should contrast this with like
an exhaustive tree like in tic-tac-toe where you could say like you know there's there's nine actions and
then eight and then seven and six and so it's a sort of like nine factorial uh sized um tree um the monte
carlo tree search and go is very very sparse right it only considers the paths that you've expanded
children nodes on okay so um now that we have the search algorithm that applies the value function as
well as the policy function um we can now talk about how the monte carlo tree search algorithm can
actually act as a improvement operator on top of these guys here 20 years ago jane street's data
center fit in the corner of an office ron minsky who co-leads the tech group there told me about how
it all got started one of our compute clusters we called the hive and i remember the first version of
the hive was literally like six dell boxes stacked on top of each other at the end of the row and the
trading systems themselves we also had there because we actually wanted the ability to make sure we could
turn the damn thing off i mean there were ups and downs like literally at some point you know one of
the people who was cleaning the office unplugged one of the trading systems in the middle of the day
as they were vacuuming so you know in the end it is in fact better to have it all in a data center
gene street's data centers have come a long way since those six tells and i got to tour one of them
in texas with ron and dan fontocorvo who leads gene street's physical engineering team you know these
cabinets these gb300 cabinets consume at peak about 140 kwe compare that to traditional air cooled you're
talking about 10 to 40 kw it's a lot more we got deep into the details of running one of these data
centers things that i had never considered before it's filled with a liquid a mix of distilled or
deionized water and propylene glycol 25 of propylene glycol that's to inhibit any bacteria or algae growth i
don't love the world where we have to worry about bacteria growing in our servers i got to see way
more of what actually happens in the data center than i've ever seen before jane street was willing
to literally pull up the floorboards and take out the racks and take me to the back where all the chillers
are you can check all of this out at janestreet.com/thorkash or we posted the full tour okay so um we now
talk about the rl part of like how this thing gets stronger by playing itself right let's say we play a
game where at the ai so you make a move ai ai will will kind of compute the search and then this is
this sort of visit count distribution um let's say this is your policy your policy initial policy
recommendation at the at the at this node and then after mcts it uh gets more confident about one of
these actions right and and so maybe the the distribution looks a bit more peaky like this
based on the the search now of course you can tune the search process so that it ends up more diffuse but
that's probably not a good idea mcts should get more confident about specific actions uh than others
but it of course might place a lot of weight on you know other actions initially and then as you increase
the number of sims it should converge to a very peaky distribution um so so this is your new uh let's call
this like pi let's wrap this in like a mcts operator of you know a given s right so after applying mcts
process your your policy recommended distribution looks like this it's a bit more peaky than the previous
one um and so then you take the arc max or maybe you just sample from this uh it doesn't have to be
the arc max and then you make your move and then um and then you throw away the tree and then you begin
a new on the next move right so again like you um you know you compute a new distribution
so initially maybe your guess looks like this and then you refine it through mcts
there should be one more x on the board right i'm sorry that's correct yes
um to something that looks like
this right so so on every move you have your initial guess from your policy network
and then the search process that combines your policy network and your value network arrives at a more
confident action that you take and and then so and so forth and then the game ends and one person
wins and one person loses so a um the way that the beauty of of how alpha go trains itself is that
it actually can take this final search process the outcome of the search process and tell the policy
network hey like you know instead of having mcts do all this you know leg work to arrive here why don't
you just predict that from the get-go right like why don't you like you know not use this guess and
just predict this to begin with and if you have this guess to begin with in your policy network
then mcts has to do a lot less work to to get things to work and so if we draw like a sort of test
time scaling plot um so so let's say like this is like number of simulations um let's say you know at zero
simulations your your sort of implicit win rate is like um is like i don't know here and then and then
um without any simulation if you just take this raw action that this is what your winning rate is and
let's say as we increase the number of sims maybe maybe you kind of have a win rate that looks like this
right so um when you search for let's say a thousand simulation steps that gets you to a um a policy
here that gets you to here which is great but if you were to distill this mcts policy network back
into your sort of uh shoot from the hip policy network then you could actually um uh you know start
here like if let's say this was you know zero um for uh by distillation then if you spend another
1 000 sim steps then you actually kind of get to here it's almost like if you could just you know
um amortize the the first 1 000 steps actually into the policy network instead of the search
process then you can begin at a much better starting point and then get a much better result
for for your uh for the number of sins that you put the save more type nature of test time
scaling as the number of simulations increases the the increase in win rate is uh smaller is that true
even for the distilled network that is to say is there some gain of like okay we start from the
distilled we get these early gains again or is that just inherent to like the nature of yeah mcts to be
honest i actually don't know the test time scaling behavior of mcts simulations and i believe it might
actually be quite sensitive to how strong this one is in practice i'm just drawing a monotonically
increasing function that gets to one okay so yeah so don't pay too much attention to the shape of the
curve just know that it's monotonic with respect to um okay so um so so the idea of mcts is very
brilliant which is like we're gonna we got something better by applying search yeah and um we're going
to now on our next iteration of updating this network just train this to approximate the outcome of a
thousand steps of search yeah and so instead of starting here we get to now have a neural network
start here and then and then you know the the play gets stronger once we then apply another thousand
steps on top of it and you can keep going right so so the training algorithm for alpha go is to
basically take the games where you've applied the search on every move that the policy encountered
whether you want or lost and that's quite important um and you're just going to train the model to imitate
the search process so um there's an analogy to robotics actually which is the dagger algorithm um
first i'm going to draw like a schematic of like let's say you know the states right so s0 s1 s2 s3 so
let's say you know we took a series of actions in an mdp to to get a trajectory
and these actions may be suboptimal right maybe we lost at the end of this game
so there is a family of algorithms that basically take trajectories and relabel the actions to better
trajectories so maybe a better action here would have been to take you know a0 prime a better action
here would have been to take a1 prime and yet another one like a2 prime a3 prime so um
what mcts is doing is basically saying like you play this game where you eventually lost but on every
single action i'm going to give you a strictly better action that you should take instead it does not
guarantee that you are going to win but it does guarantee that you know if you take these tuples
as training data so that you retrain your uh your your policy network to predict these ones instead of
these ones you're going to do better and this is very related to dagger in robotics and imitation
learning where you want to collect a intervention here and even if you're in a not great state for
example like a self-driving car that you know veers off the side of the road there is still a valid
action that kind of corrects you and brings you back yeah okay so um pedantic question but is there
a guarantee that mcts must be better than the policy for example you could imagine early on in training
because mcts is informed by the value network yeah early on in training uh when the value network
hasn't been well trained on finished uh games um that like mcts is worse than sort of randomly
standardized policy so is it just like a heuristic that mcts is better than policy or is that like
is there some guarantee right in in practice it is a heuristic um and it does work in also in practice
but let me illustrate a example where mcts can give you a worse distribution than your policy network
so um and this can often happen if your self-play algorithm um has trained to a good point but then
somehow it's uh it collapses because it's um it's not trained on diverse data or something right so um
let's say we have a board state where um the policy recommendations here are very good so so like you
know pi of as is like great but um somehow because maybe we're playing on a lot of games where the the
bots just resign instead of playing all the way to the trump taylor resolution they kind of forget um how to
evaluate those kind of late stage plants right like in the in the case that we showed with the corner
play maybe like 100 of our training data in our replay buffer has lost examples of how to evaluate
the value function at those states so you might end up in a scenario where your terminal value um is like
very bad and if the terminal values of the leaves are not good then this will actually propagate all the
way up and cause your um your your puck selection criteria and your backups to be off and then you
end up visiting a very very different distribution than what your policy initially recommended um also
if your number of sims is low then you might also have a variance issue where you just don't explore
enough right like like a it's only guaranteed to converge when you kind of take n to infinity um so so
variance in you know your search process as well as inaccuracies in your evaluation can definitely screw
with the quality of your policy network information and so that's why it's not a guarantee to improve
but and that is why i think i suspect why alpha go lee had the uh playouts to the end in their training
algorithm so they could ground this thing in real plants yeah um in practice what you could also do is
just like for 10 of the games you you prevent the bots from resigning and you just say like resolve it to
the end so you get some training data in your replay buffer to really resolve those kind of like
late stage playouts that normal human players would would kind of uh not play to yeah yeah so um so this
is why mcts kind of if you assume that the value functions are correct uh why it gives you a better policy
is because yeah and it's a very critical you know chain of assumptions assuming that this is accurate
then uh your search process should give you a better recommendation than your initial guess right okay
so if you have a cold started policy uh if you have an office your type thing yeah really what's
happening for the first few epochs is the policy is kind of useless and what you're really just doing is
hey but let's play full games and uh once you have a played full games for the preceding moves
we'll have labeled who won who didn't win and the loss for alpha zero has two components which is
like how good is the policy relative to mcts and how good is the value prediction relative to who
actually won the game from this move and this is this sort of like you can think of this being applied
to every single action or every single move right and really what's happening in the beginning of
alpha zero training is just like we're trying to get the value function to actually predict who will win
the game if you're if you find yourself in this state and you're this player uh and functionally
that's all that's happening and later on once that's well trained now the policy is also improving correct
okay one trick i did find to be pretty useful and this is not a peer-reviewed claim so just like take
this with a grain of salt it's like i found it useful in my own implementation to do to do the following
you want to first make sure that this is good before you invest a lot of cycles doing mcts right like
like it doesn't really make a lot of sense to do search on garbage value predictions um so so you want to
kind of start at a good place where this works alpha go lead does a very good thing where it just takes
human games and then you you like uh train on it and it just works right totally works um you can also
take an open source go bot play against itself um generate data also works uh so so if you have some
like offline data set that um that has realistic good play you can easily learn the late stage um value
functions pretty well and that's the that's what you kind of need to start the search process sorry can
you just read the sentence sure so so um it's quite easy to evaluate a late stage go go game like when
almost all the pieces are on the board like it's almost like a decidable problem right because there's
the lower and lower uncertainty as to like the depth of the tree so um most games played to the end by
reasonable people um will be good training data to train a good value function at terminal parts of the
tree got it okay then as you play more games the um the the search will back up good values into the the
sort of intermediate nodes of the tree and then like as you increase the amount of data your your value
head gets a good intuition of like what is a healthy board state versus a not healthy board state yeah
that those are much more subtle to judge in the mid game than the beginning or the end so the most
difficult part to score is like not the beginning or the because the beginning is just like obviously
0.5 and then at the end it's like pretty obvious who's winning so so the hard part that you want to
learn in the value function is like who is winning in the middle and so this this is actually very
analogous to td learning yes and there's a beautiful connection to td learning that we can you know
talk about in a bit uh as opposed to you know contrasting with monte carlo tree search so um so
you first want to get good value functions and expert data can kind of give you a quick shortcut i
recommend for you know practitioners just do that first just to you know initialize to a good starting
point and then if you want to do the alpha zero thing or uh or katago kind of tabula rasa learning
um then what you can try to do is on a small board play random games just take a random agent and if you
play like uh you know 50 000 games you'll actually learn a pretty good value function as well because
on a nine by nine board there's actually um you can see enough of the common patterns with random play
and then if you train a model that kind of can train on both nine by nine and 19 by nine data
um and katago was a was a proposed one of these architectures then um there's some pretty good
transfer learning from the value uh value head evaluated at nine by nine to the 19 by nine right
because this this unlike other games has much like a very much a sense of like there's not like a new
kind of piece that is introduced when you increase the size or something if we take it to its limit
and consider like a very tiny like four by four go board yeah like if you play 50 000 games you're
going to have a lot of end states that look like human play right like it's just like tic-tac-toe at
that point um so so if you like broaden this a little bit to like nine by nine five by five or nine by nine
it's not unrealistic to imagine that like purely random play will actually generate pretty reasonable
looking boards and then so you can score those pretty easily and so that is what gives you the
bootstrapping to be able to then improve your policy with search but it's very very critical that mcts has
accurate value uh estimates yeah and you need to ground the value ultimately mcts will fall apart
if you don't have a grounding uh function for um for the value i'd be curious about how much compute you
save by training the value and policy on the same network that because they share the same representations
how much more efficient learning is because that would be interesting if um they're basically kind of
i we've just talked about how they're kind of making similar predictions or they should be in line
with each other yeah and so i'd be curious if like actually yeah you just you're you're like
having the amount of compute you had to do by giving them the same network right alpha go lee the
original alpha go paper had two separate networks yeah and then um in all subsequent papers um they
they merged them into two heads and presumably this saves compute but answering that question in a very
rigorous scientific way is actually uh it's a simple question but but in practice actually takes like if
you really want to chase that question down to its its limit it's uh it takes quite a bit of work to
you know really resolve that yeah um but intuitively yes they share a lot of representations so so and you know as we
mentioned there is there is a sort of like your your policy network and your value network when doing
evaluation should kind of agree right so that there really should be this sort of consistency between
them yeah probably this is the wrong way to think about it i feel like when i learn how an llm works and how
simple rlvr is at least as an algorithm how simple it is i'm sort of stunned by the kinds of things it can do
uh that it can learn how to build very complicated code repositories and whatever simply from getting
like a yes no and here i feel like if you understand it more deeply of like just predicting mcts and
it actually seems awful go seems less impressive in retrospect the more you understand it because
you're like oh you're putting in a lot of bias by just saying how much you do you're like telling
it how we should titrate exploration as things go on you're building this very explicit uh tree search
for it and so i don't know if you show that intuition where it actually the more you understand the less
impressive the accomplishment in 2017 seems i uh i personally disagree i i think they're profound for
different reasons and i don't understand the lm rl like enough to like kind of comment on your podcast
about it um but um i think alpha go so so yeah why is it a profound accomplishment i think maybe it's
worth stepping back a little bit and just like uh it is different than modern rl and we can talk a little
bit about like some of the algorithmic choices there but um i think the most profound thing here is that a
um a 10-layer neural network pass so basically uh 10 steps of 10 steps of reasoning yeah and of course
the reasoning is not just one trail of thought it could be like the distributed representations and a
lot of thoughts going on at the same time but uh by construction let's say a 10-layer neural network
can only do 10 sequential steps of thinking right um 10 steps of neural network paralyzed distributed
representation thinking is able to amortize and approximate to a very very high fidelity a
nearly intractable search problem yeah so so this was a breakthrough that i think most people don't
even understand today like fully comprehend like how profound that accomplishment is and this is what
also girds like um alpha fold for example right like um uh where you have a very very difficult
physical simulation process that you would need to roll out so many micro scale simulations and yet
like 10 steps of a somewhat small neural network can somehow capture what feels like a you know np
class uh problem into a single um problem and and so i i it actually makes me wonder if you know our
understanding of problems like p equals np or you know these very fundamental like computational hardness
problems are incomplete right like like it's it's not like you know obviously this is not a proof of like
p equals np or anything but but there's something to it that like kind of is very disturbing where like
what felt like a very hard problem can fall to a very very simple macroscopic yeah that is a very
interesting insight that a lot of problems which are proven to be np hard like i don't know if go is
proven to be np hard but okay protein folding etc have been like neural networks can solve them because
they're np hard in the worst case but we're not dealing with the worst we're usually not concerned
with the worst case we're you know like these problems have a lot of structure to them yeah i think
that the the the kind of question we should be asking ourselves is like we've been formulating you
know solutions to np hard problems as in like kind of worst case complexity and i wouldn't say you
know this solves go right it doesn't give us a exact solution of the optimum yeah but in practice like
it is extremely useful and the same thing has been shown in like alpha tensor alpha fold where like
yes there is a very hard problem that in the worst case seems intractable and yet we're able to make
like almost arbitrary amounts of progress so so here's a sort of like uh you know in the limit what is
what what might this look like right well um if you want to simulate you know something very complex like
weather or um predict the future like you know do we live in a simulation or not um the computing
resources you need to build a very complex simulation might be much smaller than you think based on you
know our ability to amortize a lot of that computation into the forward pass of a single network interesting
so to me alpha goal was the first paper that kind of like really showed this like profound level of you
know simulation being compressed into a small amount of um i feel totally not at all qualified on the
complicational complexity of the math to comment on this but i wonder if um there's an important role
of chaos here where if under what is the problem with weather and why does it take 10x the amount of
resources to predict whether a day out uh and continually so for every more day out is because it's a chaotic
system and so small perturbations can totally change the final estimate as time goes on and um i guess
it's interesting well i guess you would expect that for go and protein folding as well so here's an
analogy to whether that might be relevant and go so um the problem of like you know here's our current board
state yeah um given what we know about both players what is the board state in the future yeah what is
the exact board state in the future right this is uh this is extremely sensitive to initial conditions
like a single stone place here can kind of disrupt the entire prediction yeah right so this is hard this
is kind of intuitively the chaotic problem um and yet somehow so this is this is hard somehow we can
predict who's going to win yeah like and this captures a lot of possibilities here and so that
there's this more macroscopic quantity that we really care about which is the average or expectation or
some sort of global macro structure over a lot of like you know possible futures and so so in whether it
could be the same thing right like we don't exactly care like what the you know velocity of wind 6 000
feet above a specific latitude longitude is we kind of care like where's the hurricane or you know you
know things like that and and i would say like in chaos you know there's a classic like lorenzo tractor
which kind of looks like this right um yes you you don't if you start anywhere on the lorenzo tractor
you don't know where you're going to end up but you you do know that the thing looks like this yeah yeah right
and and so there's this kind of beauty of like sometimes we don't necessarily care about the
micro scale things we actually care about the macroscopic structure interesting and and these
things can be predictable and contrast that say to something like a hash function which is also incredibly
dependent on initial conditions but doesn't have a macro structure at least hopefully if like the
arguments work yes um and so there's like no equivalent of a value function or like broadly how's the weather
going to be that is interesting there it's really just about what is the move what is the board going
to look like 100 moves from now exactly uh yes intuitively that seems correct i i uh and then again
this is also out of my my area of expertise but i um i find it interesting that like cryptography has not
been able to like um the tools of cryptography and uh you know um hashing have also not been able to
prove that like uh you cannot come up with fast approximation like you cannot come up with fast
approximations right like if that if they were able to do that then you could prove p is not equal to
mp yeah yeah in fact we know that there's structure and many cryptographic protocols obviously like uh
rsa cryptography there is structure and that structure is what quantum computers exploit
to break them right i see um reiner has a very interesting blog post which we talked about in the episode
where he uh talks about how if you look at at a high level what cryptographic protocols look like and
what neural networks look like it's extremely similar where you have sequential layers of jumbling
information together and it's because there's this convergent evolution in the algorithms where in
cryptography you want the final state to be incredibly sensitive to initial conditions so that
it can come out sort of looking jumbled based on if you change anything and then neural networks you
similarly want everything to be dependent on all the information because you want to process all
the information and consider how it relates to itself yeah you have the maximum power of a neural
network at the edge of chaos um i think there's some like research papers from uh joshua schuldig's
yeah yeah like like there's something kind of quite fundamental about like chaos that is it's not just
like hopeless noise it's like there's something kind of useful right in in um in chaotic systems at
least at that boundary um but yeah this is just my like think about this philosophy i don't i don't actually
know the math uh well enough to comment on it anyway if we go back to um we'll talk about lmrl a little
bit because there's some connections there but let's just go back to like the mcts like what is it doing
it is not um crucially it is not saying we're going to uh increase the probability of winning
directly it's not going to say like we're going to um up weight all actions that won and down weight
all actions that didn't win yeah um importantly what it is doing is saying for every action we took
we did a pretty exhaustive search uh on mcts to see if we could do better and we're just going to make
every action that we took better by predict like having the policy network predict that outcome instead
and and so this is um a very very nice idea because you have one supervision target for every single
action yeah so the the variance of your your learning signal is very low compared to the alternative
naive rl thing so let's actually consider what let's consider a very naive algorithm uh that looks a lot
more like you know modern lm rl today where where we do something like um let's take the winner of a
self-play game and encourage it to do more of that okay so uh it's worth kind of thinking a little bit
about like okay what are some alternatives if we could do to train self-play agents instead of mcts
right like you know we use a lot of uh lm style rl these days like is that relevant could we do that
instead um so let's think through this a little bit let's suppose we have a very naive algorithm where we take a
league of agents of different checkpoints and we play them against each other and um for the games where a
single player wins uh we're going to reinforce those actions up and then and then uh and train retrain
the policy network to imitate those those guys instead of uh instead of the mcts objective um
so what ends up happening is let's say you have a chain of actions that led to a win
and you're you have a matchup between two agents that are basically the same so in fact let's just
assume that like uh you know policy a and policy b are like evenly matched right so they're true
their true win rate is is like 50 percent um so let's say you play 100 games
and then each game let's say lasts you know 300 um moves
and um you're doing some sort of like evolution strategy or some way to perturb these things to
get them get them to do different things or maybe you don't and you just play them against each other
and you see like occasionally this one might actually have a better strategy than this one right and so
so let's say um you know 51 games um policy a wins
and then 49 games policy b wins and this is just due to random luck or maybe you perturbed policy a in
some way that let it do this and just to have a very very simple model let's pretend that for
for like uh 49 of the games they played exactly equally um i'm sorry for for 50 of the games they
they played exactly equally right um and on that one game where this one won it played slightly
differently it made like one critical move that like you know normally it would have done differently
but due to some exploration or some random noise it just happened to make a smarter move than it did
previously so you have one supervision signal like one true supervision signal for your policy network
and then you have uh 99 games times 300 moves for which imitating those actions gives you exactly the
same policy you had before and so the um the scale of your variance is actually very bad because it's like
you only have one label out of this enormous data set of of actions of supervision actions where you
want actually sorry let me let me clarify a little bit okay so we're just talking about how the good
move the outer distribution move is a small fraction of all the moves that are played across all the games
on which you'd want to train and um this of course reminds me of how llms are trained with policy gradient
methods uh karpathy was when he was on the podcast called it like sucking supervision through a straw um
and uh and so yeah it's interesting that this like this thing you're saying which would be intractable
and prevents you from actually getting beyond a certain level in go is just by default how llms are trained
question mark right so um in this case this is not to say it doesn't work right like if you imagine
increasing the number of games to like you know millions of samples you actually can get some meaningful
supervision like samples so long as you find a way to sort of mask out the supervision from these guys
and then this is where things start to get pretty related to rl in terms of advantage and baselines and
so yeah so so let's um let's look at the you know the gradient variance of a very naive approach like this
where um i'm just going to call it like gradient rl and it's basically the you know sum of rewards
all right so so the sum of rewards is the return right so so like uh in in our naive setup here we
only have an indicator variable for the return where either you want or lost so
um so in the case where you lost well you just don't train on your gradient is zero you don't
train on those examples and when you won you try to predict those those things right so you can
think about this setup as a as a special case of this general formula here um the um the trouble here is
that this is very high variance because when you multiply these terms out when you take when you try to
compute the variance of this and so so variance of the gradient is equal to expectation of squared minus
and just for simplicity we can pretend this is like you know on average zero or something
if you're centering it at you know no signal um and uh the variance here basically means that you're
you know taking the square of this product term and so you end up with a term that kind of grows
quadratically with the with t so so variance um when you have a setup like this this thing acts as a
coupling effect on top of of these terms here so um um let's actually map this to an llm case and we can
answer like why do llms only do one step rl instead of a multi-step rl scenario in llms you have a decoder
that might you know predict some words like hello world and so in current lm rl they treat this entire
sequence as a single action just a t and big t is just one right and so yes it is true that you know the
because of how you know transformers are formulated uh through the sort of product of conditional
probabilities um we do have uh you know probability of this sequence is equal to the sort of sum of
log probability of the whole sequence is equal to the sum of the probabilities of like you know
individual tokens right so so in this case i would um i would say something like you know log l
plus log low plus log world so um this is true
and if this term were one then they would be the same thing however um in sampling things if you have
a reward term assigned to every specific token now you have these interaction effects between the cross
multiplication of these terms and these terms right and so the problem becomes how do you ascribe the
credit associated with every episode to all these different terms here i guess the thing i'm confused
on is what would that even look like to do it that way in um in llms because you do you only do get a
reward at the end of the episode so you could imagine a reward that says like i'm going to give you some
process supervision yeah uh where you get a reward for each of these actions on every step okay so you're
saying it instead of doing it that way where you um well i guess the way you've written it would be a
sum at the end anyways so it would they wouldn't have to be multiplied but you're saying instead of
doing it that way you would just add up this process rewards at the end and then treat that as one
single reward signal correct for one single log action but um isn't that how it's written to begin with
anyways like the sum of the uh rewards so so the the thing that's a little bit hidden here in the math
is that we're assuming that when you decompose the problem to a multi-step problem that you're now
introducing kind of correlations between your actions through the computation of this guy
and uh so if you separate these things out then there will be um this this will magnify the variance
of of this one so in in the case where you don't separate it out if you just have t equals equals one
you just have a single estimate of log prob and a single estimate of reward now there are this this
term still shows up in l so in lms it looks a little bit more like the naive reinforced estimator
looks a bit like return of the single action plus times you know
it looks kind of like this this is sort of the very uh basic form here but this is still a
contributor to variance um so you want to make sure that like you don't uh similar to how in this case we
were training on a lot of neutral labels you want to make sure that you're subtract you're sort of
penalizing the labels that don't help and only rewarding the ones that actually make you better
right right so intuitively the analogy here is like can we find a term in our training objective
such that it's actually kind of discouraged from doing this or you know these don't have any effect
on the gradient and this has an effect on the gradient right i i guess if you apply that there
the only thing you could do is eliminate 49 of the games so at least the way you have it in there you
would be um 51 times actually the optimal case is to pull out discard all of these moves and only
get a gradient on that single move that you got better yeah but how would you do that right so this
is a pretty tricky problem in practice and so this is where advantage estimation happens in reinforcement
learning so you want to subtract um you know a term from from um from your your multiplier instead of an
indicator function of like one and zero you want something that kind of behaves like a zero for all
of these guys yeah and then a one for all these yeah but you so you could do that if they're um
if you can say hey i won in this game so this is slightly above baseline performance well you won
on a lot of games exactly but but you don't know which ones let you win because they were truly better
right versus winning on access how would you design a baseline where it's truly better yeah so this is
where in rl people use things like td learning to better approximate the quality function the queue that we
mentioned earlier so you can try to subtract that from your um your your your return i see so ideally what
you really want to do is in rl you want to um push up the actions that make you better than the average
and um push down the actions that make you worse than the average and they call this advantage there
are multiple ways to compute it i highly recommend john shulman's general advantage estimation paper as like a
good you know treatment on how to um how to like think about various ways to compute it but the at the
end of the day you know um you you want to reduce variance by trying to make this smaller and so it
doesn't magnify the variance right makes sense so but um this requires you to have a very good estimate
of what average performance from a state would look like so and this is this gets back to the value
function thing we're talking about earlier right and so this uh keep in mind that in this case this
model free rl setting uh is trying to solve a credit assignment problem where you don't know
which actions were actually good and which ones were bad monte carlo tree search is doing something
very fundamentally different which is it's not trying to do credit assignment on wins it's trying to
improve the uh the label for any given action you took and so we can actually think about a
completely different algorithm called neural fictitious self-play which was used to great effect in
systems like alpha star and um and open ai's dota so so let me talk a little bit about how um how you
can kind of unify some of these rl ideas in the model free setting as well as the self-play setting
okay so what happens if you don't have the ability to easily search a tree right like in go it's a
perfectly observable game you can easily construct a pretty deep tree that completely captures the game
state in a game like starcraft where you don't have really complete control over the binary it's a little
bit hard to do this and i'm not even sure if it's a it's a deterministic game right so so that makes
this uh kind of difficult from a data structures perspective so um what is done instead is that the
basic idea of supervising your actions with a better teacher is still there right so so if you know given
neural fictitious so we're going to talk a little bit about how neural fictitious self-play works
the same idea we're going to like come up with better labels for each of the actions we took just
like in mcts but how do we derive the better labels
in mcts we perform search to and assuming we have a good value function the search will kind
of give us a better result than our initial guess in in a game where you can't easily simulate a search
process what they do instead is train what is known as a best response policy
so you fix your opponent so let's say you're you're currently training pi a against
a strong opponent pi b in starcraft maybe like you know those are the zergs and you're playing protoss or
um so you fix your opponent and you treat this as a classic model free rl algorithm where your goal
is just to beat this guy and so here you use your standard td learning style tricks or use ppo or any
actually like you know model free rl algorithm to try to hill climb against winning this player and
so um you train you train basically you you have a reward function that's like you know return is like
you know one if wins against ib so this is no longer a self-play kind of problem right this is
just like a fixed opponent and you're just um solving it trying to maximize a score against against that
and then you know zero otherwise and so you're you have a sort of fixed environment where all you
care about is just beating this guy and um once you have a good policy that you train with uh you know
pick your favorite model free or algorithm ppo or sac or you know any kind of mixture of the or you
know uh vmp or whatever um you now have a good policy that gives you a good label for what this one
should do when playing against that player and when you train multiple best response policies you can
basically then distill the rl algorithms into the labels for a given opponent so you might have let's
say a best response policy against pi b and then maybe you have a collect a league of you know um of
opponents like pi b pi c pi d and you're going to take the best response policy that you train against
each of these fixed opponents and for this one you're going to uh supervise them with the label that this one
would provide so it is kind of like this is almost like a proxy for your mcts teacher right instead of mcs
teacher you use a model free rl algorithm to find the best search action that you could do to uh to to
kind of beat your opponent and then you're finally you're distilling the um the policy here into what
is known as like a a mixed strategy where it's trying to basically average across all possible
opponents you could play against and this is what gives you something that can do no bet no worse than
like you know an averagely average selected opponent from the league and and so this gets around the
problem of having to derive a teaching signal from mcts but it's still fundamentally is about relabeling
your your your your states with better actions so that they improve your policy and just make
sure you understand this is like if you win against against against this other policy you sort of
reinforce all the actions yes on that trajectory yes so here you can use a number of algorithms like ppo
vmpo you know q learning even if you want like uh the specific algorithm here um can be you know
it's usually a model free thing because you don't have search but it is an interesting connection from
mcts and q learning that i i want to you know bring up so in mcts you do something where you have a tree
and through the resolution of your your value function at the at the leaves of the tree or
you know your approximate leaves of the tree you can kind of back up through through the you know the
the sequence of many sequences and then obtain some sort of mean value estimate right but your q
is kind of derived from the average of a bunch of simulations in model free algorithms there is often a
component uh of estimating a q value and so um a q q values are often learned through td learning although
in ppo the the way that the advantage estimation is not necessarily through a bellman backup but um but
in q learning there's this kind of a very cool trick where um you do you know q s a is
backed up as r plus you know some discount factor times the max a q of your next step so intuitively how
this works is like if you have a mdp
and then this is like you know terminal
what this is sort of saying is that like the best action you can take at
this state is equal to the reward you take for you know taking this action plus the best that you can do
at the next state so there's a sort of recursive and dynamic programming property of mdps and you can
train neural networks to basically try to enforce this this const this uh consistency right so you can
say like well once i know the q value of this action i can then use that to kind of compute something
about the q value support so when earlier i was like hey why are we training policy why don't we just
train the value alone that that is what this is um this is a algorithm for recovering value estimates of
intermediate steps when you don't have the ability to do forward search so you must collect a trajectory
first of like n steps before you're able to do this trick um but the intuition is kind of the same
which is that like knowing something about the q value here can tell you something about the q value
here and indeed you can recover a policy from a q value right so so the um you don't need to
explicitly model the policy distribution you can actually recover the policy distribution by doing argmax
over your um your q values right so so q q learning or you know this kind of like a approximate dynamic
programming kind of propagates what you know about the future cues backward like this right and you can
see that there's a sort of similar structure that goes on here where uh in in this case you're planning over
trajectories your agent hasn't actually been to yet whereas in this case you're planning over trajectories
your your agent has visited um so so importantly why does q learning you know why was q learning a big
deal right like it's because historically we just haven't had the ability to do search on fairly high
dimensional problems like robotics or whatever so for a long time we kind of make the assumption that
like okay well if we can't model the dynamics with like a world model or something we're going to
instead just collect trajectories and then plan with respect to the only number that really matters which is
reward okay so this is very interesting and then to unify this with our discussion of llms so with llms
you're doing something you don't have q values but you're doing this sort of backwards learning where
hey let's find the trajectories which pass some unit test in some coding environment and then let's reinforce
those trajectories and then there's a huge difference between that and this forward approach with mcts
and the reason you can do mcts and it's much more preferable to mcts because you can do it per move
and make each move better rather than having to learn per trajectory um and hope you know as karpathy
said hope to learn this like straw yes you get the supervision through a straw uh basically just upgrade
all the tokens in a trajectory that might or might not have been relevant to getting the answer right
the reason you can do this much more sort of sample efficient uh much more favorable thing with go
is that because mcts works in go you basically know that hey if i just do search locally here
and this search is sort of truncated at the end by this value function that uh works even even if i
haven't unfolded my whole trajectory i can just say this is my new policy um and i can improve in a more
iterative like local way rather than having to having to unfold all these trajectories so there was some
research i think from google in 2030 2023 2024 where they did try to apply tree structures to reasoning yeah
and i think it's you know the jury is still out as to whether this can ever work so i would say like
uh it we probably will see like you know revisiting of this idea of forward search um in in the future
but there's two things that make mcts very simple for go which is that value estimation is kind of
concrete and you can determine it for real and then you can kind of uh uh sort of use it to truncate
depth as you said yeah and then the uh breadth is also uh determined and what's kind of critical is that
the action selection algorithm where you iteratively visit and grow the tree is um well suited for the
size of problem that go is and the depth of the problem but for something like lm reasoning um you
know pucked might actually not be a good enough heuristic it might be too greedy with local tokens and
it might do something like oh only give you you know uh sort of obvious thoughts that are correct but
not really solve your final problem yeah so i would say the jury is probably still out on how like
what the final instantiation of reasoning for llms would look like and i wouldn't rule out that like
this stuff could you know come back but it's been hard don't llm sort of night and natively learn
to do mcts where they'll try and approach and be like oh that doesn't work let's back up let's try
this other thing and then go in the direction that proves to be more fruitful uh yeah certainly i think
that llm's managed to do something that looks like real human reasoning without having to do an
explicit tree structure yeah um that being said i think the idea of doing forward search and simulation
to get a better sense of what is valuable might make a comeback um even though not exactly in the same
instantiation as as uh but uh just to make sure i understand the crux of it like the the breadth from
the number of legal actions being wider and the depth from being able to not being able to train a value
function as easily because so here's an example where lms break down the c puck rule involves you
know square root of n over one plus n a in an llm like you're most likely never going to sample the same
child more than once right so if you have let's say multi-steps of thinking um because language is so
broad and open-ended it's a sort of uh discrete set of actions is not really an appropriate choice for an llm
even though they're discrete tokens um it's just such a large number that this type of exploration
heuristic is probably not the right thing to do to guide how to search down a tree right but i i guess
the crux comes down to the fact that in go you know that the mcts is almost certainly better than
your current policy even though you haven't gotten even though you haven't explored the end of any
trajectory correct and then in uh in normal reasoning for lms robotics there's no way to just locally
evaluate and improve your next move in a way that doesn't result in in a way that's independent of
actually like solving the problem no way is a strong word i think lots of people have thought about how
to try to apply mcts or its kind of successors like new zero to continuous control spaces and i'm sure
you know very cool research work is still ongoing to try to crack that problem um but yes the the
seeming challenge right now is that like most problems in much higher uh dimensional um you know
action spaces or something that's combinatorially much bigger like language they they don't seem as
amenable to the kind of discrete action selection heuristics as well as uh kind of game evaluation
type stuff that uh go does um but that's not to say the idea of like you know thinking into the future
along multiple parallel tracks might not give you some information about like which way to search
right like if you think about mathematics i think mathematics often occupies a little bit more of
like a logical search kind of procedure where you kind of can back up you can kind of see like which
paths seem good or not there's more of a rigid structure there whereas maybe like in a uh you know
business negotiation or something um it's less of a tree and maybe you know something a bit different
okay so we're now seated so i can ask you some more questions about awful go and about
ai research more generally um in 2021 andy jones had a paper called scaling scaling loss for board games
and um he basically anticipated inference compute or inference scaling by showing that you can trade
off test time compute and uh training compute that is to say that you can spend more compute on the
for the searching through the mcts um and if you do that you can get the equivalent performance as having
spend more time training the model um and so if you you know you if you see this pattern you might think
okay well with llms you might do something like that in the future in fact that's what had ended up
happening okay so what is a kind of fun exploration one could do now to explore other axes of scaling
in toy settings which will be important to understanding what ai development might be like in a few years
sure yeah um i think that indeed test time scaling and uh reasoning and how it interacts with model size
are quite profound when it comes to like how much uh needs to be actually done as explicit search versus
how much can be packed into the forward pass of a neural network right and and um how does a four
pass of a neural network sort of learn how to do something that should be a sort of sequential and you know
recursive step that's quite interesting yeah um so the yeah the andy jones scaling laws for board games
paper is quite cool there's another really nice result from that paper where they where he showed
that um not only can you predict scaling loss of like you know the sort of llm variety where um
as you increase parameters you can decrease the amount of compute for search or vice versa
he also showed that you can actually predict uh how much compute is needed to solve a
larger version of the board game uh for example and and so with go you know which can scale from you
know uh three by three to infinitely sized you know uh go board you might actually be able to sort of
revisit this question and try to reproduce uh whether this shows up um you know i actually started this
project with this sort of a motivation that like does the bitter lesson or does our knowledge of
scaling laws allow us to kind of execute a lot better on a sort of compute optimal go bot and can we
can we kind of build a strong go bot without all the kata go tricks right just just by really
focusing on the bitter lesson the scaling laws um i have not been successful so far but i think it's
it's sort of a fact that like usually when you want scaling laws to work you want to be in the
regime where um the the recipe already works and the data sets are good rather than trying to kind of
figure out how to do scaling while also trying to figure out what the the right data set are okay so
so this is like the scientific understanding component in research often follows a step where you get
something to work first and then you use that um system to collect data that then helps you
build a mental model of how things work such as scaling laws right and and so usually actually
you want to build a strong gobot using scaling laws you actually have to make a strong gobot first
and then use the scaling laws to kind of extrapolate a bit farther into the future say more so just so
i understand first of all you're saying scaling laws did not work or you could not there was
no scaling loss pattern that you could see in your gobot yeah so um a mistake i made initially
when i had some bugs around how mcts labeling was working was i would um i would collect a bunch of data with an expert policy and then
treat it as a supervised learning problem and try to identify scaling laws with uh expert data sets
you can indeed plot things that look kind of like this but if you're in a regime where you know your
policy is not working well you might be just studying scaling laws on like bad data right so so just like
one important implementation details that if you want to study a scaling laws problem you kind of have to
have a problem for which like the data is good the architecture is good and there's no bugs and then like
you you you solve it there um ex ante i wasn't able to like apply scaling laws to direct what to look on
look look at um until you know i had read the rest of the system working and this sounds obvious like
researchers of course you want to have like a working bug-free system before you study scaling
but uh just as a sort of advice for practitioners on like where i actually tripped up when i started this
project was you don't necessarily want to kind of jump into the science of studying your man-made
artifact before your man-made artifact is like interesting enough to be studied speaking of
compute so if you can look at these charts of compute used to train the best ai model in the world over
time going back 10 years and it's a very smooth line in log space uh that is exponentially growing year
over year uh except there's this huge aberration and that aberration is often goes zero uh which is
train on way more compute than any other ai model at the time it was like uh three e23 flops it's sort
of comparable to like a frontier llm i mean orders of magnitude off but still um and so yeah the question
is especially with you being able to get something off and did you train it on your own i got a donation
from prime intellect okay for like about 10k and then i spent um i spent maybe the first 4k doing um kind of
exploratory research yeah and then uh about 3k on the kind of final run yeah um and then uh some some of
it remaining for serving the model cool yeah is your sense that they were just did a badge up for reading
it if you can do it in 10k now uh the compute required to be the first to do something is always
like much larger than the compute it takes to catch up and it's the same story playing out in lms right
like once someone else has done it um you could use tricks like distillation you could use um uh all sorts of
like kind of uh crutches to kind of bootstrap your way to success so with my own bot that i've hosted
online um i actually used uh sort of best response training against the katago models to kind of get
a strong level performance and um you know as a time of recording i'm i'm validating whether this
can be uh i can kind of do that first step which is to do the tabula rasa right um but importantly for
research you often want to start from a good in it right so so the kind of simple thing i did first
was train best response agents against katago yeah um alpha zero team uh they did not have any
policy that they could train against right because they were trying to do everything tabula rasa so
um and being the first to do it means that you're prioritizing getting the thing working rather than
like let's say the most compute efficient uh possible implementation so um this actually plays out in
robotics as well like if you look at the kind of frontier of large models trained for robotics um
there the scatter plot is all over the place and there isn't a very clean line the way that there is for
frontier lms and and that is because the folks training these models often are not you know at
the scale where every flop counts and they need to kind of squeeze out you know the performance of
every single flop as the dominating decision deciding factor in pre-training right instead their focus is
more like we want a certain capability to to show up so we optimize the training setup to kind of make
it easy to derive that capability and once you have that capability well um invariably if you scale up the
compute you are forced to kind of make it compute efficient because this is like hundreds of millions
of dollars we're talking about but um but in the past when when compute for experiments was kind of
more plentiful or you know not not uh not um uh accounted in a way that the researcher was really
responsible for then you kind of end up with people optimizing for things besides kind of being on the
compute optimal preto frontier i see like speed or something yeah like time to result or just getting
to work i think the first alpha go like probably they had lots of compute and they didn't need to be
they didn't need to worry too much about making it the most compute optimal yeah and how much of the
improvements to compute efficiency are methods that did not exist as of 2017 versus things which you
they could have done in 2017 but um yeah great question so so going into this project i kind of knew
in the back of my mind that like things always get easier to do over time and i want to see like where
where where is go at given that like it didn't seem like there has been any major open source you know
strong bot it's after katago in 2020. and then you know reading the katago paper there's a lot of
clever ideas i was kind of wondering like okay uh let's look let's see if the bitter lesson has happened
where like a lot of these kind of tricks just sort of go away because the nvidia made faster gpus right
and and so uh roughly where are we on that so um again this is not a peer-reviewed claim so this is
just my preliminary um you know vibe guess on like what i've seen based on my own experiments
but it seems like um you know architecture choices don't matter that much you know transformer versus
resnet we're we're at the sort of speed of gpu where the size of the model is not so big that this
really matters um you can actually simplify this setup quite a lot so instead of doing a distributed
asynchronous rl setup with replay buffers and pushers and collectors you can kind of do a a dumb
synchronous thing where you like collect you just train a supervised learning model and then you
collect again and and so there's like opportunities to simplify infrastructure um nvidia gpus have indeed
got faster so whereas katago was trained on v100s you can train on like half the number of you know
desktop blackwell gpus and it still works um and um some of the kind of auxiliary supervision objectives
that katago developed aren't really necessary if you have a strong initialization right so if you're
initializing against you know best response training against katago itself then your own model actually
needs none of the tricks that katago needs yeah so so then the core thing is like how can you get as
quickly as possible to some strong opponents and um that matters a lot more than the specific
architectural innovations but there are still some nice compute multipliers so i found that training on
nine by nine boards was very nice for resolving end game value functions and then like if you can co-train
that on a architecture that can transfer between nine by nine and 19 by 19 then you can really cut
down the worm start time to learn that from scratch i think alpha goes zero their plot was um first 30
hours or so are spent basically catching up to the supervised learning baseline um and you can cut down
that time a lot by kind of pre-training on a small board and then and then like you know worm
starting that into your you know 19 by 19 board play uh there were some other stuff like you know
varying the number of sims between episodes this turns out to be not that sensitive actually like
you can kind of you know fix it or increase it doesn't matter too much um but so anyway it's kind
of just nice from a scientific perspective just revisiting like an old paper and seeing like what
really matters this is sort of a potential question but why is it okay to have a buffer and off the go
because every time i talk to any research here they're telling me about how bad it is to be off policy
but then the way a naive implementation of alpha goes zero would work is that most of the moves in a
given backward step or at a batch of backward steps um would be not not among the ones that were made
by the most recently trained model so why is that okay great question yeah and this this gets into the
sort of fundamental off policy versus on policy uh reinforcement learning kind of questions so uh as you recall in
mcts you take actions that you took and you relabel them to uh to take different actions on the same
states right so so the off policy part here comes where um what if you're relabeling states that your
new policy would never visit right like what's the point you're kind of wasting capacity and in the
extreme limit imagine your distribution of states in your training buffer are all states that you would never
visit then you're basically supervising them uh to uh take good actions on states you would never achieve
and therefore your policy can get really bad right so this is where off policy can really hurt um uh alpha
go um however if you interpret this sort of from like the dagger perspective which is basically saying
like a way to kind of correct yourself back to the optimal trajectory uh given some some data what what
you kind of want in a algorithm like this is to have mostly states that you would visit but then you have
a small percentage or maybe a reasonable percentage of states in this kind of high dimensional tube around
your optimal you know trajectories and any of those states are given a supervision target to kind of
uh sort of funnel you back into your optimal trajectory so maybe i can just draw quickly here great so in
um sort of a dagger style setup what your kind of optimal training data distribution is is that
here is your optimal states and actions so this is like you know you want to be in this state you
want to be in this state you want to be in this state and then you win here um and then these are your
optimal policy actions so these are the these are the things that you definitely want to train on
but to make it robust to disturbances um you want to make sure that if you happen to drift off
into some other states you can kind of funnel yourself back into but why isn't this a fully
general argument for our policy training this is actually why you want to do off policy training
sometimes is that like you you don't want to have a compounding error where if you make a mistake
you don't have the data of how to return back to your optimal distribution yeah and so um optimal
control does not really say too much about like uh you know how to uh you know not accidentally get
here because it's sort of making the assumption that like once you learn the policy you're gonna
get here but in applications like robotics right like like i don't know a gust of wind blows you
slightly off and then now you need to like correct right um or the friction on one of your tires
is kind of a little bit like lower than the other wheel and then now and now your car is drifting and
you gotta like correct it so so these kind of things in in like more real environments often happen
where like um actually there's a funny uh quote about chess and also go it's like the problem with
uh the problem with go and chess is that the other player is always trying to do some shit right
like uh so so like you know things can kind of drift off yeah and you always want to be able to
correct uh back to your back to your winning condition so so your replay buffer really should have like
your you know the states that your policy would visit plus some distribution of states that you might
drift to and then how to return back to your optimal states yeah now if you take this to the extreme
and you say like well let's uh we don't have any of this data
and we're gonna just like be labeling with mcts um you know states that are so far away from our
optimal behavior like this this bag of states over here well like now yeah i mean like each of them
gets mcts label and your policy learns how to do take sort of the best possible action here but you
never get here so like you're training your model on states you would never reach uh like like this is
this is not there so then this is a problem right and this is where off policy can really hurt yeah um
so actually as part of this project i did try an experiment where i took a bunch of trajectories and to
try to saturate the gpu as much as possible what i did was i took uh you know random states from the
data set and uh re-ran mcts on just those states right so instead of playing a whole game where i'm
doing mcts on every move i just ignore the sort of causality of moves and just pick random board states
and i just label those with my current network and uh and i might revisit old states that i've labeled
before and relabel them again with my current network right and so in practice this actually does work you
can actually say like let's take some states that are reasonable and constantly be relabeling them
in uh in uh in um while we're training and so this actually starts to converge on a very robotics-like
setup which is very common which is you have your your data set of trajectories um and then you have
something like a replay buffer pusher
and these are off policy offline trajectories right so your replay buffer pusher pushes transition tuples
to to to the replay buffer
and then you have some job that's kind of continuously um replanning
what the best action you should have done instead of taking this action is right and so in robotics it's
actually very common to use a a that sort of minimize td error so like your bellman updater
constantly is pulling things from here and trying to satisfy you know the qsa
so so um and then and then from here you have your trainer
which is trying to fit the s to a or or or uh um fit the you know q to the q target so so here you can
think about this as a sort of planner right you visit old um states that you've been to and you take
your current model and you rethink like what could i have done better if i visited this and um and so
this is actually how like kind of off policy robotic learning systems are usually trained um these days
there's a sort of simpler recipe but but like you know in the google qt op days we kind of did did
things like this so what is the trainer oh yeah the trainer is uh you try to you try to minimize
uh qsa and qtarget which are going to explain the whole setup again like at a high level yep so you
have your off policy data that came from various policies you're constantly pushing uh transitions
that you saw before to a replay buffer no and then you've got this thing called a bellman updater
which basically replans instead of this action what action should i have taken at s to have a better you
know value and the way you enforce that is you try to minimize the td error so so actually you given this
you have s prime right you you compute q of s prime and you find the action that should go with s prime
that makes this q value as high as possible and then you add that to the reward here and that gives
you your actual target right so for this current s and a your q target is this so now you have a now now
you send back the q target to to this this transition so with this tuple you pair with that a q target
and then here on the trainer you simply just you supervise learning and you minimize your current
network's qsa with its target got it okay so in the background you're just like hey let me basically
think through how valuable were all these actions actually yeah in a more optimal policy where you're
trying to maximize this what is the q target of this transition it's sort of like basically daydreaming
exactly yeah you can think about it's like you're kind of going back in hindsight and being like
like like given what i've seen in his historical buffer um like was there a better action i could
have taken yeah now the connection to go here that i tried and it was you know moderately successful but
too complex to kind of like open source was um you replace this with like a mcts relabeler
where um instead of doing this kind of target network computation you uh run mcts on your transition
right so in in this case you have uh your state your action and then whether you want or not at the
game um and actually you can just toss these two you don't you don't care about these ones you just take
your state and you just plan mcts to get your best policy you know pi
on your current network right not not the network that took this action but your current best
policy network you just rerun your search offline on these transitions and um if these are transitions
that your policy can get to then this actually acts as a very nice stabilizing effect and also the one
other benefit is that you can like kind of fully saturate your gpu better because you're not like
blocking on the go game to kind of like give you board states you just simply search across all board
states at any depth in peril yeah so and then here the trainer would be just you know predict the mcts label
as possible so so again like this kind of works and this is quite relevant in robotics where you're really
you just have i'll have a lot of offline data and you can't simulate things like mcts but in practice like
it does run into the problem where you know like if the current model is looking at states that it would
never reach then it's kind of wasting capacity yeah and so you have to be a little bit careful here
so um the on policy thing and also much of rl has kind of converged to a much more on policy setup
where they don't really try to like directly train on off policy data at best they use off policy
data as a way to reduce variance but not directly influence the objective hmm i'm sorry why have they
conversion that it's just more stable okay yeah yeah so so like you might use the off policy q as a way
to do like you know advantage computation um like you know q minus like sum of q you know that's kind of
like your your or sorry like you know sum of uh like there's n actions and then yeah so so like uh
so like this is your value and then this is your your kind of current q values your advantage for that action
is like the average value minus your current one so so like people can try to estimate q in an off
policy way and then like just use advantage here and then and then the the sort of if there's a problem
in these dynamics that it doesn't like blow up your loss as much um and so in robotics there's a kind
of convergence towards more like uh using off policy data to just shape your rewards but not actually be
directly here i'm reminded now of our earlier conversation of why mcts is so favorable as
compared to the kind of you know reinforce a policy gradient kind of thing lms do and this might be
totally wrong but i wrote a blog post a few months ago about um how rl at least policy gradient rl is
even more uh inefficient than you might think and so the inefficiency one thinks about naively is the
fact that you have to roll out a whole trajectory in order to get any learning signal at all at all and
so as these trajectories become longer and longer as an agent has to instead of just previously like
complete the next word in the sentence it has to go instead to hey do two days worth of work to figure
out even if you even did this project correctly the amount of information per uh flop has been
decreasing as you had to unroll two days worth of thinking in order to see if you even did something
correctly to like did i implement this feature the amount of samples per flop has been decreasing but so
you can think of um uh you're trying to maximize as you're learning bits per flop right um
and this is you can think of bits per flop as um samples per flop times um uh bits per sample
and what i just mentioned uh a second ago is that the samples per flop go down as rl becomes more and
more along horizon but um at least this kind of naive rl is also terrible from a bits per sample
perspective and here's what i mean at least compared to supervised learning so early on in training let's
say you have a uh vocabulary size for an llm that is 100k long so there's 100k possible you know tokens
that one could answer and you have a totally untrained model and you have a prompt like the sky is um
with supervised learning you what would happen is that the model would have some probability distribution
over all the things it could say um there's a label that says actually the term here is blue
and it would figure it would learn basically for cross entropy loss exactly how far its distribution is
from correctly saying blue now if you're doing this through rl um you would say the model would try
the sky is halicon nope that's wrong the sky is told nope that's wrong this is a totally untrained model
right and so you would have to do this on the order of a hundred thousand times in order to just stumble
on blue then get some learning signal off of that so if you're in the supervised learning regime and you
just get you have your distribution of probabilities you get told uh that it's blue and you figure out how
far off you are the amount you learn is um is a function of your pass rate so like the further away you
are from blue the more you've learned to go towards blue uh using cross entropy loss and so you can think
of it as like your pass rate your like prior probability of having said blue and um as a function
of that like in supervised learning uh through cross entropy loss you would you would learn negative log
p p p being pass rate uh bits once you get this label whereas in rl if you're just randomly guessing
shit and seeing if it works or not that's um that's just basically going to be the entropy of a binary
random variable which is and what's also tough here is that actually the distribution that you're sampling
under is your policy's distribution right so so it's like if your policy has no chance of sampling blue
then you will never get a signal exactly right so that's that's being modeled by the fact that
your probability of sampling blue is extremely low if you do sample it you do learn as much as you would
have learned in a supervised learning in all other cases like you know 99.99 of in an untrained model
you're um you're just learning incredibly little from like seeing how the con is not the correct word or
tool that's not the correct word um and that's what happens most of the time so you're just like
um learn very little so if you try to graph um if you put on the x-axis your pass rate um
and uh here you put the like sort of the bits you bits you're learning from a sample if you have like
like zero percent here 50 percent here and 100 here so the end of trading you're here um if you have um
supervised learning negative log pass rate would look something like this and then the uh entropy binary
random variable would look like this um and this is uh depending on whether you're doing nats or bits
uh yeah if you do bits it's like one right here at the at the peak um this is like a coin flip you
learn the most from a coin flip uh this is supervised learning this is rl however the problem is you spend
most of training in this regime right like in the in the low pass rate regime and um in fact of how fast
you're learning is a function how many bits per sample you're getting uh and you're getting very
little signal here if you chart the pass rate on a log scale so you put the x-axis on a log scale where
like at the beginning of the vocab size of 100k the pass rate is 101 over 100 000 then one over 10 000
one over 1000 uh one over 100 and then um okay what this graph looks like here where supervised learning would look like this
and
and then rl if you just basically crunch what i just showed there it would look like that yeah
and arguably you spend all your time here potentially never even getting a single success
right exactly like uh so so it's it's a sort of depressing plot in the sense that like once you're
here it's not at all obvious how you get to here yeah um you know once you're here you have something but
like you actually in many rl problems spend all the time here yeah uh so so there's a sort of question
of like how do you initialize so you're at least not at zero but like at a non-zero pass rate yeah um
one more thing i'd like to add about this per sample that's very relevant to um uh you know any kind
kind of machine learning problem is that um it and there's a connection to soft targets and distillation
where if you have access to the logits right not just the one hot like this this is a sort of one
hot uh token answer yeah um if you have access to the soft targets um the entropy of this distribution is
far far higher than than the one hot so there's actually way more uh there's way more information in
bit and bits per sample um in a soft label yeah so that's why distillation is so effective yeah
per sample is that it's actually giving you way more information person yeah well i wonder what the
equation would be but obviously it would just be the entropy of this distribution like so the entropy of
this is zero yeah um the entropy of this is like you know the entropy equation and this is also why like
you know alpha go is quite beautiful in alpha go you don't train the policy network to imitate the mcts
action you train it to imitate the mcts distribution interesting but both of these are actually valid and
if you wanted to do a scientific experiment of like how important are this kind of soft label dark
knowledge distillation you can run an experiment where you you uh retrain the policy network on the
action mcts selected rather than the software interesting earlier i was sort of stumbling around
this intuitively why is this ability to do um iterative search where you don't necessarily need to be able
to win the game in the beginning you just need to be able to improve your current policy why is that so
powerful a capability in learning as compared to how llm's currently run our learn rl and um and yeah it's
exactly this thing of uh this is considering your pass rate of the entire trajectory i actually don't know a
a formal way to think about this maybe you should help me out here why is alpha go an elegant rl
algorithm yeah like so um uh the major reason is that you never have to initialize at a zero percent
success rate and solve the exploration problem of how to get a non-zero success rate and this is what
allows you to hill climb this beautiful supervised learning signal where and if you look at the actual
implementation of alpha go every step of the way there's no there's actually no you know td error
learning or dynamic programming at least explicitly it's just supervised learning on a value classification
as well as a policy you know kl minimization so it's just a super supervised learning problem on improved
labels and so the training is very stable right you can train like as big of a network as you want you can
kind of retrain this on the data set everything will just go stably the infrastructure is very simple to
implement as well um you don't need a complex distributed system to kind of keep everything on
policy um at the end of the day you're just saying like i have some improved labels let's retrain my
supervised model on these targets yeah and and so you're always in this beautiful regime where you're
just trying to improve the policy rather than uh escape this kind of like uh sort of local minima where every
every signal is flat all around you yeah um so so one way to draw the the curve is like if you draw the
sort of win rate of an mcts policy versus the raw network um let's say that's dotted line is the raw
network the mcts policy kind of looks like like this and so every step of the way this supervision
signal is very clean right you're never in a situation where you know the mcts is kind of like giving you
no signal yeah unless your mcts distribution converges to exactly what your policy no yeah yeah okay that's
that's a great way to explain it um cool okay maybe we sit down and i ask some questions about automated
research sounds good one thing i really wanted to talk to you about is that you did a bunch of the research
for this project through this kind of automated uh llm coding assistant loop and um there's an idea that
if you fully automated ai research you could have some sort of singularity uh obviously we're not
there yet but to the extent that we have early indications of what this process might look like i am
curious what your observations about um what the ai is good at what it's not good at what you think about this
scenario it's likelihood eventually what thoughts you have about this in general for sure yeah i think
automated scientific research is one of the most exciting um uh skills that you know the frontier labs
are developing right now and i think it's important for everyone who's doing any kind of research to
get a good intuition of like what it can do now and what it can't and how might the sort of science
process work in the future once we're having ai's automating a lot of this this investigation um so
in brief i mostly use opus 4.6 and 4.7 throughout the working on this and um what works is that the
models can do a very good job of doing hyperparameter optimization so in the past people would kind of
come up with a search base of hyperparameters like learning rate and you know weight decay and maybe how
many layers are in your network and um they would just kind of do a grid search or a sort of bayesian
hyperparameter optimization uh approach and then it would find some tune parameters um the kind of really cool
thing that automated uh you know coding can do now is that it can search a much more open-ended set of
problems right it can say like well um i've identified that like the gradients are kind of small in this
layer so let me change it up here let me rewrite the code so the data later data loader has a new
augmentation i came up with let's uh let's uh sort of try to find the best way to kind of fit the
constraints of the optimization problem and and you end up with this much more flexible and kind of high level
almost like grad student like uh ability to just you know grind a performance metric and and so this
can squeeze out quite a lot of performance you can you know on a fixed data set with a fixed time budget
um improve perplexity by quite a lot on on a sort of classification problem like lms or um or go
and uh it is also fantastic now at basically executing any experiment right so i have a
clawed skill that i wrote called experiment where um i give it a description of what i wanted to
plot and like i just described here's the x-axis i want here's the y-axis answer this question for me
and it'll go run off and do all the experiments compile the plot make a report and suggest like
you know what might have caused it or or so forth um so so that's what works quite well today and i think
we can expect that these abilities get better in the future but it's also kind of useful to know you
know what what is it not doing so well today um so on my blog version of this tutorial i have a
a plot of basically all the kind of experiments i did grouped in a sort of tree where um you know
every node kind of represents a failed successful or sort of mixed experimental result and then from
there it branches off into a child where it's like the follow-on experiment um occasionally i'll kind
of rabbit hole down a track like this off policy mcts relabeling do a few experiments and then realize
it's probably not worth it so then i'll kind of jump to a completely different track right and i call these
kind of things like rows right so so what i find is that current uh you know closed models that we can
access the public can access today um they don't seem to be that great at selecting what the next
experiment should be in a given track and they don't seem to be able to kind of step back and do the
lateral thinking of like wait a minute this track doesn't really make sense like let's go back to sort of
first principles and and think about you know what the bottleneck might be or like what are we trying to
achieve right and and so often i had to catch infra bugs myself by prompting the right question to
clods like investigate you know why why what is causing this discrepancy and then it'll answer the
question um i think with like you know mythos class models or mythos plus plus models coming online um
maybe this just completely changes and these these problems just fall to to just improve skilling um
but at the same time i think there's a lot of like rich opportunity to um develop rl environments that might
incentivize this kind of lateral thinking and and so one of the motivations for setting up this go
environment was that i think that you know go captures a lot of very interesting research problems
often overlapping with you know lms or robotics and yet it's like very quick to verify um the outer loop
is ultimately like does the agent do what i think it does and and you can kind of check the outcome of
a go game quite easily um and then the inner loop involves all this kind of like you know research
engineering around distributed systems uh predicting whether an idea is going to work or not um predicting the
you know the difference a particular modification to your training algorithm might make um and i think
there's a rich library of subtasks and sub environments that you can kind of train an automated scientist to
work on uh with go as a sort of outer verification loop that then once you acquire these skills maybe you can
apply them to like other domains like uh you know um biosciences or robotics or automating ai research
which is which is the real crux or the um scary slash uh incredible thing of just making ai's making
future versions of ai's and you're suggesting the outer loop here could just be your win rate against
katago basically um that's one of them um i think there's a lot of deeper questions that one could
tackle right so for example um let's say you have an idea on um how to improve a scaling law compute multiplier
yeah um the outcome isn't necessarily like i i uh achieved the best go bot ever the outcome might just
be like can i predict what the win rate of my go bot will be yeah or can i predict the scaling law plots
that emerge from my idea but then you can verify that you haven't kind of reward hacked anything by using
a very verifiable game like go on the outer loop i think there's a couple of interesting follow-on
questions there's questions on the inner loop in the outer loop on the inner loop there's a question of how
locally verifiable any modification you might make is that is to say could would you know whether
something is actually improvement or a degradation some idea you try out would you know that if
something isn't working as a result of um a bug or is it the result of the idea itself being wrong
um ilia was talking about why having one of the reasons he he thinks he's a good researcher is he is a
good researcher one of the things he thinks makes him a good researcher is that um he has intuition
about he has strong belief in what the correct idea is and he is able to persevere through bugs and know
which things are bugs versus mistakes in the fundamental idea based on his high level belief
about this idea should work so therefore it's this there has to be bug versus the other way around
why don't we start with that question actually yeah how locally verifiable are things which are good
ideas yeah i think as in the case of the success story for deep learning you can think about this as like a
decades-long idea that took like took a lot of faith to get it to work um and so this presents a very
challenging long horizon you know rl problem where you know every every step of the way you have like
a committee telling you that this is a bad idea and then ultimately you break it through right
and so like how do you design rl environments that maybe give you some feedback uh uh earlier um and
and and i think this is a very tough open question that i don't have an answer to but um but you know
ultimately to play a very strong gobot you probably did need to discover deep learning yeah right and so
i think that like having a challenging game that cannot be you know cheated easily on the outer loop
could be used as a sort of outer loop signal for something like discovering the principles of deep
learning now of course like to make it tractable and this is where research tastes really matters
like you have to come up with ways to initialize your problems so that you don't
solve a sort of very intractable problem right like maybe you can leverage lms as a sort of a universal
grammar in the middle to kind of give you some sort of local feedback um um the fact that lms are
universal grammar means that they can kind of move at almost any level of the stack right they can think
very locally as well as step back and think like in very broad steps and i think that's where a lot of
uh um the the lateral thinking ability of humans kind of come from like like how to know if the track
that you're pursuing or the objective that you're pursuing is not right and you should be
asking a different question the uh the other question is how stackable local improvements are
in the attempt to get to a better result on the outer loop um i've heard rumors that at some
ai labs the thing that has gone wrong is that people will individually pursue good ideas
um but those don't end up stacking well and so the training run falls because of some weird
uh interaction between two seemingly good ideas and having a single top-down vision of how things
should work is very important um having worked at uh different ai labs and also playing around with
i guess parallel agents trying different ideas what is your sense of how parallelizable
um ai ai innovation is yeah great question um
i think the research taste for executing well on you know the bitter lesson is that you need to know
how much the bitter lesson can buy you and uh how much is too much to ask for at any given moment right
like of course in the fullness of time compute kind of is the single most important determinant on like
how things work and uh and and uh it's almost like inevitable that as you scale up energy and compute
and parameters intelligence will just fall out of that and that's super super beautiful super profound
no algorithmic detail really matters beyond that but um in present day we don't have infinite compute
and parameters and and an arbitrarily good initialization so we have to come up with like
heuristics that kind of give us that but these heuristics are probably somewhat redundant so that's
probably why you see this effect where like a lot of these compute multipliers don't necessarily stack is
that like they might have some correlated benefit um and then and then you know three years down the line
when the nvidia gpus have gotten even stronger maybe maybe they stack even less well right like maybe
like at any given point in time the the sort of benefit of any given compute multiplier is transitory
which is what i sort of suspected with the katago paper like there was many algorithmic ideas kind of
applied and then you can see that like with you know modern blackwell gpus and ada class gpus that are
much better than the sort of v100 uh grade gpus that that paper used um you can see that like some of
these algorithmic tricks to speed up convergence just don't matter so much compared to something else
and i think that's a matter of taste in a pre in the present time yeah interesting how about the outer loop
how verifiable for making ai smarter with go you do have this outer loop of um win rate against the best
open source model out there and even there as you were saying there are other outer loops of did you
discover a new phenomenon which is actually very hard to if you didn't know scaling laws were important
if you're back in when was chinchilla or kaplan scaling was released like 2019 yeah so if you're back
in 2015 would you there's not an automated procedure one can easily imagine of uh knowing which paper is the
scaling loss paper versus which is just like another random plot and so that even in the go case is hard
to verify outer loop and the whole whole the whole idea of an outer loop is to have like some backstop
on um on improvement uh but let alone for general agi where of course we have a bunch of these benchmarks
but there's a problem that like we know the things we can measure and we improve on the things we can
measure but we care about this broader ability to do economically useful work which is um at least
until you automate everything easy and not super easy to measure um so yeah there's there's a question
of okay how how good is the outer verification loop for uh for ai self-improvement and does that matter
yeah um i'm gonna give a non-rigorous argument but one that i kind of intuitively believe which is that you
know um deep mind the ai research lab um they started as a sort of focus on games right like they they kind
of use games as their outer loop and then their researchers learned uh from experience of solving games
and then like now they're working on lms and presumably there was some positive transfer from their
time working on games and like atari and go and and uh you know starcraft that like now helps them make
good lms like i assume that there's like positive transfer in some regard whether it's coding or general
research ability or project management right like all these things kind of like probably help them do
well um and so if that's the case why wouldn't it also be true for automated ai researchers like they
should be able to positively transfer experience tackling quick to verify uh quick to iterate on
environments to something more ambitious and economically useful like uh you know automating drug discovery
or so forth i mean i don't know if it isn't the it hasn't the issue with uh historically until
gemini through or whatever been a couple years ago people were saying look google hasn't uh
isn't catching up in llms because they're too tight to the old approach and yeah there's gains but
there's also um there's ways in it which actively hinders you um so it's actually not obvious to me that there's like
like the jury's still out right like i think like who knows if the you know let's say currently google's
doing quite well who knows if the uh initialization on training on games is ultimately going to hobble
their ability to be the winner in the long term right like like uh it's it's hard to say for sure
and uh you know likewise who knows if the late seeming late start was really just them kind of
pre-training for longer on how how to like scale up tpus right they invested all their tech tree in
like uh getting tpus to be good which seemed not that useful in the short term but then long term it
becomes maybe like a yeah so it's it's even hard for humans to reason about what the optimal research
strategy should be yeah even with the the data we have today yeah yeah cool um okay we should let people
know how they can find out more about this project whether to fork it themselves whether to check out
your blog post we're doing an excellent job explaining many of these ideas um uh where do people go
next great yeah so my my website is uh evjang.com the there's a blog post that kind of links to a
interactive version of this tutorial um and on my github uh which is the username is just eric chang
uh there's a there's a auto go repo that people can fork and reproduce the uh training results
and i also highly recommend people check out this blog post as rocks may think which we touched on
some of the ideas in this conversation but it's this uh grander uh you know um thesis of what happens
when you have uh thinking as a primitive in computer science exactly right um and so i highly recommend
people check out that blog post as well yeah and i encourage to the you know the audience to you know
think about the relationship between thinking and go you know via mcts and search and how it relates to lms i
think there's something quite like profound there um and probably underexplored just because go has
been relatively underexplored compared to you know the boom and lms you know um it's not to say that i
think we should have trees in our in our lms but um but but but there is some very interesting duality
between them and you can actually do a lot of research on go um mcts and reasoning with you know very
small budgets so that's very exciting cool awesome eric thanks for doing this it's an honor to be on the podcast