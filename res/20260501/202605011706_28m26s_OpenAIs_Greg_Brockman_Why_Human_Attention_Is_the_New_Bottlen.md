---
title: OpenAI's Greg Brockman: Why Human Attention Is the New BottleneckOpenAI's
uploader: Sequoia Capital
channel: Sequoia Capital
channel_url: https://www.youtube.com/channel/UCWrF0oN6unbXrWsTN7RctTw
duration: 1706
upload_date: 20260430
webpage_url: https://www.youtube.com/watch?v=bBS93A0BeNI
id: bBS93A0BeNI
categories:
  - Science & Technology
---

# OpenAI's Greg Brockman: Why Human Attention Is the New BottleneckOpenAI's

so Greg thank you for coming back here I don't think we ever charge you for rent so maybe I'll
send you an invoice later but Greg you've been part of like two really spectacular companies
stripe as employee number four and then the first CTO I just recently heard that they process 1.6
billion sorry 1.6 percent of the global GDP you must be proud of that that's amazing you must be
even more proud of the fact that opening I has almost a billion or maybe more than a billion in
weekly active users at this point I mean it's all it's all very exciting it shows you what technology
can do and you're not just co-founder and president but you're also a chief builder at opening I heard
that was one of your titles I'm not sure there's ever an official title but I've been called many
things let's just say that well you have a audience of great builders here so we'll start from all the
way at the bottom of the stack you open AI has multiple stacks of the business one of which is
compute and you guys have been very aggressive very aggressive on securing compute why is that well in
many ways we have a very simple business we buy rent build compute and we resell it at a margin that's it
as long as the margins positive then you want to scale it because the demand for solving problems the
demand for intelligence that's unlimited and the AIs that we have right now really are able to rise to
the challenge of effectively any kind of problem that you want to throw at them do you have enough
compute no really yeah definitely I was just with Matt Garmin and he says the GPU compute availability
in 2026 rounds to zero don't you guys have all of it I mean we have we would love more we're
constantly out there hunting for more honestly and I'll tell you like when we first launched when we
launched chat gbt I remember being on a call with my team and they're like alright how much compute should we
buy and I said all of it and they're like no no seriously like like come on how much we buy I'm
like no matter how fast we try to ramp compute I guarantee we're not gonna be able to keep up with
demand and that has been true ever since that's that's fascinating moving up from compute since I
don't know if much of this audience can help you with securing more compute because most of them are
founders of startups about architecture and scaling laws are what are the what are
this where are we in the scaling laws are they still doubling each year are you
changing architecture what's what is what are you guys pushing on the frontier on the
research side well I would say first of all the scaling laws are a deep and very
beautiful mystery right they feel deeply fundamental it's like the scientific truth
that just like you think about physics and you know Newton's laws and things like that there's
somehow this truth of the universe and they're empirical like we don't necessarily have all the
theory to explain exactly why it works but to me the most beautiful thing is that neural networks
were really designed like in the 1940s before they were computers and somehow we've been able to take the
exact ideas that were developed back then and apply increasing amounts of computation and
as you pour more compute into the models they get corresponding more capable and it just keeps going there's no wall
and that's I think that's a beautiful thing
that's pretty beautiful are there more research or more algorithms that are in the works that because you know in the past we had neural networks to your point in 1940s but we couldn't we
we didn't have to compute for it now that we have to compute for it you just or are we just pushing the same things or are there new
architectures and new ideas coming up yeah so I would I would think of it as we we absolutely have new ideas that are constantly powering what we do it's very simplified to say well let's take a neural network from the 1940s and you know put put in a
in a gigawatt data center right we have made tons of innovations and we constantly are improving things and sometimes these are micro tweaks like you just realize that the wave and formatting data was not quite right and that can actually be a very big deal
sometimes it's larger you think about the shift from the LSTM to the transformer and I don't think the transformer is you know like everyone's moved past the transformer as described in the other 2018 paper so there's there's constant innovation happening and I think of places that have been perhaps the most invested in long term
research on how to improve the architectures how to improve the fundamental algorithms and how to get the paradigm shifts I think OpenAI has been leading the pack there and that's something we continue to invest in I see lots of fruit on the horizon
and on the models does OpenAI have a formal definition for AGI are we close or are we not close Pat and Sonia published this thing that we are at AGI functionally do you agree with that do you not agree with that well we do have a formal definition but to some extent
I one thing I have learned is that everyone has their own intuitions about what AGI is and maybe you can view it as like according to my view of where we are I think we're about 80% of the way there
and that we have models that are smart they're very capable they're able to if you give are they smarter than you
I mean they're certainly more capable than I am at writing software right if you give it all the context then yes I think that they are they're just so capable it's really remarkable
Like does anyone here feel better at writing software than gpd 5.4
All right writing kernels. I'm so even there we're seeing massive gains from
Exactly where's and for some of our internal results that there we're really seeing if you pour the right kinds of of
You know if you have the right setup for your problem
Then you're able to get really massive results out of very low level
Even low-level tasks and just to give you one example of how things have been trending
One of my systems engineers also very similar was like hey like I haven't been able to get value out of the models for gpd 5 or 5.1
For 5.2 as well for 5.3
He on a lark had prepared this design document for a very complicated systems optimization. He was about to do
he
Handed it over to the model went to sleep waking up intending to like give this to his team to work on for the next week
And we woke up. It was done that the model had actually
Implemented the initial spec had seen that it was slow had added instrumentation had actually run the code
Used a profiler to figure out where things were slow and iterated multiple times until it got into an optimized result and like like that is incredible
That's where we are and so how what would you advise all startups here to do because the models keep getting more and more capable
they're kind of
I've asked this when Sam was here in the past and you know what if you're building today
Do you need to rebuild in two years when a new model comes out because all the functionality and all the capabilities all change around you?
Do you need to make sure that you're not in open AI's way because
You're kind of wrong. You're just gonna run over startups because the models are so much more capable
How would you recommend a set of
Startup founders to to build in this environment?
Well, first of all I would say to lean in the tools right now have become incredibly useful and if you look even over the course of December
I think that we went from these agentic coding tools being like
You know they're like writing 20% of your code to writing 80% of your code
Which means they go from being kind of a sideshow to being the main thing that you're doing and I think we're doing that across all of
The work that people do with computers all computer work this year and you can look at the recent progress on codex
It's really changing from a tool for software engineers to a tool for anyone who's doing work with a computer and
Just over the past week we've released a bunch of features that just make it so much more powerful and capable
and
One thing we just announced today is a new tool called chronicle that plugs into the codex where it actually can see everything you're doing with your computer and
Can form memories of what's going on and so you ask it a question you just it instantly knows what you're talking about
You're like, huh, what was I doing five minutes ago? It knows right? You're like, oh, what was this person talking about?
It knows it's to me
It was this real wake-up call to realize you spend so much of your effort right now
Just explain these your computer what's going on like why are you explaining to your computer? What's going on?
That makes no sense and so I think what's going to happen over upcoming years is the models are going to get much more capable
We'll have better harnesses will be able to be able to solve harder and harder problems come up with new knowledge all of these things
But there is a one-time shift that's happening now, which is really about context is really about
Is your AI able to you have all these meetings you didn't include the AI?
You know, that's not very nice to the AI like you're asking it to help you with things and it has no information
so I think really leaning into how do you make sure the AI even has enough information in theory to solve the problem and then trust the models are going to really get there and improve so
I think it will be a constant cycle of improvement and iteration and leaning into the tools and kind of talking to your friends if you're how
They are using it, but that there is this investment. That's a one-time investment that now is the time to make and in terms of like
Let's say we you set that all up. How do you how is open AI using codex differently than you think?
Everybody else that's using it
Well, I think one of the amazing things about being at opening eyes. You do get to live in the future, right?
You do get to really see the shape of what's emerging and we can co-design, right?
We can really change the models the harness everything together in order to better serve the needs that we see and
A lot of the approach we've been taking is so we started with software engineering and we set some clear guidelines for
example saying that
We still want a human to be accountable for all code that gets merged, right?
So at the end of the day
Is it a good thing to merge this piece of code is well structured is going to make our code base more maintainable?
We want to make sure there's a human who is signing off to say yes
And that's I think that thoughtfulness of not just saying okay
Let's just blindly use this or you know
Oh, we don't want to use this at all like I think neither extreme is quite right and then we are also going vertical by vertical with an open AI to adopt these tools within
finance within sales within
IT and there we have a small dedicated team who's really deeply understanding the domain working with the people who are the experts in it
in order to build skills in order to modify the codex
UI whatever it is that is is needed in order to get it to be good and then that's something we can then
Once we have it in good shape we will externalize and that we're able to ship that to all of you
And so we are starting to work with certain customers as well
So for people who want to be very AI forward and want to be part of defining this revolution
That there's a place for that and I'd love to talk afterwards
But yeah, I think that just this desire to say hey
We really want to be AI forward really live in the future and experience what it will be like for everyone else
One year two years three years down the road
Do you guys structure your company differently or the engineering teams differently because of the living in the future?
I mean if you have to go way back when my father learned computer science
He was just himself and then we had these long software releases that became waterfall
And then when the web happened and the cloud happened we had these two pieces of teams and we had scrum
Now that we have these coding agents is it is how do you structure around everything differently?
I think we're still figuring it out and there's certain places where you really see it for example
The cost of building a prototype is cheap now
It's so cheap and if you want to build a dashboard that used to be like I take like someone like a week to do it
And you just do it now and so actually a lot of the bottleneck has shifted to things like
Sharing like how do you and so we actually have some internal work on this as well that again
We will be externalizing of how do you make it really easy for anyone in your enterprise to build a dashboard a widget a bot
Whatever the thing is and then share it with others
And then that starts to really put pressure on having good governance like you want your it organization to be able to see all these different
You know threads of execution that are happening all the little things that are being shared around
Have some control over
Data provenance right to really make sure that okay like a good example of this is um
I think people are now starting to take their internal knowledge dumps tournament to wikis
We have some some a really cool one of these internally and
The thing you immediately think about is well if someone has a document in
The internal knowledge base was accidentally permissioned incorrectly and they realize oh no
I didn't want this information to be accessible
How do they fix that right so normally it's they go into the dock they change the permissions
But now there's these derived artifacts and so you need to make sure you have some way of tracking through the system to say
Well, this output document came from this source one the source one is no longer accessible to this audience
Let's go and invalidate that as well
And so you have to start really building your technical architecture
With awareness of the way that people are going to use this information and it really changes how teams relate to each other because you can just
It really changes where the bottlenecks are and what's hard
Do you think team size is going to be a lot smaller we're going to have
Still human software engineers in a decade?
Well decade is a long time from now and that the ceiling on this technology is hard to
It's really hard to internalize I think that it is clear that what a company is will change in a lot of ways
I think that we're going to have
This ability for solopreneurs to build very incredible businesses and so anyone who has a vision I think will be able to realize it
I think the jobs that you all have will become way easier in a lot of ways way more fun
Now might be more competitive too right because everyone's gonna have these amazing tools
And so really figuring out what is your niche?
What is your unique angle is probably going to become
Kind of the most important core but a lot of how we run organizations right now and it's there's almost only one way to organize
large groups of people where you have teams your management structures and you have
Scopes and you have these hierarchies and all these things maybe that can change maybe you can be much more flat
Small teams that can really just do incredible things like we're seeing it right now in
Mathematics where these individuals on the internet are using gbd54 pro to solve these unsolved math problems and we need a math team and
They're just doing it. Yeah, my son's a math nerds
I just told him that maybe you should be studying something else besides math
But I well, I see this is the question right is if you look at something like alpha go, you know move 37 this move that just like
Changed humanity's understanding of the game
But the thing that was surprising is
It made the game more interesting and important for humans and maybe that'll be true for for these other domains, too
true
What about common failure modes when you're building your new years you're building with
Production agendic workflows what do you what do you see as the common things that founders get wrong and they're building incorrectly these days?
Well, I think that
These models they have such power and really understanding how to operate them well takes thought
And so we've been investing a lot in primitives security primitives observability having again good governance things like that
But just to give you one anecdote that I think is evocative
I asked so I was working my codex
I asked it to install some package that someone had opening I'd written ran into an error
I was like oh ping that person on slack and ask them for help so ping the person on slack
two minutes later it said this is taking too long
I've escalated to the person's manager and it actually pinged the person's manager
And and you realize it's like on the one hand it's kind of a reasonable thing for the model to do it's being proactive
It's trying to solve my problem
It's like, you know not just sitting around waiting to be told what to do
But on the other hand like, you know
Maybe you should have taken a little bit longer maybe should have checked with me and so I think that
Really thinking about these questions where we're still building up the eq of the model
And that in some places it's getting very good for example clicking approve approve approve
Is kind of where we've been and humans are not very good at that either, right?
It's like
They just default they just default and so now we're starting to have ai's that can actually take care of flagging
Is this a high-risk action? Hey, this one should be escalated this one's okay to auto-approve
And it really makes you realize that human attention is going to be this incredibly scarce resource, right?
The doing of things now is easy the is this a good thing is this what I wanted?
Is this aligned with my values with my desires that is going to become the single most important bottleneck?
And so I think building systems that take that into account and really think about the human factor like that's the most important thing to do now
another human factors security
um
how would you advise people to think about security in this world of ai and
Just heard about breaches left and right with versell recently and then
And these models are incredibly powerful at finding security holes
So how would you recommend people here use the models to find those security issues?
Well, I think there's a couple levels to the answer
I do think that this is
I think that the internet has been
A place where security has been just like a
Ratcheting important concern over time you think about where it started
Going through the 90s with viruses and worms and malware and those things and we've moved past that
I think we are also moving now to a much more ultimately secure regime
But it does require kind of an internet-wide effort to get there
And so a lot of this honestly is just again leaning into the technology having these models
They can scan your code base. They can actually be used for end-to-end red teaming
Like there's a lot that can be done with them
And a lot of how we're thinking about further models and improvements there is really leaning into
How do we how do we actually sort of leverage trusted access programs?
How do we leverage the community of people who really care about
Being defenders and making the internet more secure. I think that's something where
Everyone has a role to play and can participate
But the number one thing is just sort of recognizing that these models are very powerful
But they're not magic right that they are
Just like a part of the overall resilience ecosystem and I think that we as a society
And I think every company again really contributes to this have something to build in terms of how do we
How do we incorporate these in a way that results in more assurance and more
Sort of certainty on the impacts of of whether it's this particular patch that you're taking whether it's thinking about
How do you make sure that you're you know, just sort of rolling in updates quickly as they're being released?
So I think that there's a lot of work to be done, but I have a lot of optimism for where this is going
Um, let's switch to speed. It seems like things are moving faster and faster and faster and faster when the world of accelerating change
We were talking about it when we when you you were walking up here around health
How you're trying to keep up with things? How do you you keep up with all the accelerating change?
How would you recommend everybody here keep up with everything that's changing?
Well, I think this is the new normal and I think to some extent it's not really because of ai
I think it's just been the trend of technology for the past two decades
There's more people doing things it's easier to do things than ever barrier to entry goes down
It means it's also much more easy to build value right to have great successes
And so I think that really trying to keep your ear to the ground and understand what's changing and
To some extent it always starts with the same thing
Which is play with the technology yourself
Like it's very different to hear ai described versus to use it
But the beautiful thing about ai is it's so intuitive like that's the whole point
Is that rather than have the machine be something you have to contort yourself to
The machine contorts itself to you, right?
It's doing work for you
And it should be something where you ask it and does something
And so I think that just really trying to just get your finger on the pulse
Of what's changing what's possible where the models lag
That is I think the core skill that is going to really determine a lot of the success of of companies in the future
And then on the flip side of that you guys have held up held back models to work with security agents
So it's like the opposite of like going as fast as possible
So you're doing things responsibly too
So how do you like think about the balance because you're in a competitive environment?
You want to ship as quickly as possible and yet you're trying to do the right thing as well
Yeah, I think at a values level like what open ai is about
Like we really want to put the power of ai in people's hands like we believe that people can
We want to empower people to build the future with the tools that are being created
But we need to do that in a thoughtful way right that we really think about both sides of here are the benefits here's the risks
How do you maximize the benefits?
How do you mitigate those risks?
And I think that in cyber security and in biosecurity those are areas where we're very thoughtful
We've been building we've been working on these kinds of both mitigations and trusted access programs for quite a long time
And that what we see coming is models that are going to be increasingly powerful and capable in a continuous way across all dimensions of capability and the
You know, we announced last week
The expansion of our trusted access for cyber program. By the way, has anyone here applied?
No one. Oh, I see one hand two hands. Okay, more of you should apply. It's great
We really need help because and it's very important the people who are trustworthy and responsible and really want to push these models
Are participating in this because that is how that's going to pay dividends for everyone
We're going to have more to announce over upcoming weeks on how we're expanding the program
But and also when we release models to everyone kind of the mitigations that we have and how we're going to tune those to be
Both to really balance right to really try to bring these capabilities as broadly as possible
While also making sure that the ones that are you know that we're thinking about the risks and and able to
To have some observability over them and to ensure that this is maximally positive in terms of deployment
So I think the short answer is like it's core to our mission
We care a lot about the impacts of what we're doing not just building the technology in isolation
But it is a whole community a whole world effort to really get to where we need to be
On um now moving up from the models to the application layer, which is what a lot of people here are
Building how do you how does open ai decide what in the application layer you're going to build and what you're going to leave out?
Well, people have probably seen the word focus being applied to open ai
Quite a lot recently possibly for the first time smiling in a while
And it's been applied to her too
And it's it's hard because the field of ai is one of opportunity, right?
It's like anything you're going you can imagine is going to be great. No question is going to be great and
We as a company as a single company no matter how much compute we build no matter
How many people we have are only going to be able to do so much and so a lot of where we've been
how we've been thinking about things is what is the
Sort of most focused strategy that covers the parts of the space
You know, maybe it's an 80 20 or just like the parts of the space that we think we can have most impact on
And I think there it's very clear right now. We're going through this identity transition and so
Products that are it's not just about enterprise versus consumer, right?
So it's like clear we are being very serious about enterprise like we're selling to big companies and building a whole muscle and sales motion there
But consumer what consumer is is going to change, right?
It's kind of a very broad term that buckets in multiple things
But the slice of consumer that's about not just productivity
But about goals about achieving your goal about even knowing what is your goal being able to elicit that and having an ai that can proactively do that
It's all kind of the same thing like in the end
We're trying to build an agi that you can talk to that has all this context that you can use in your personal life your work life
That's trustworthy right that you can go to it for advice and give you useful information maybe health information or maybe about finances or
You know about if you're trying to figure out what to do with your career like all these things
They all kind of ladder into one thing and it's meant we had to make some very painful decisions about what not to do
But I think I would just say that that's the aperture that we look at things through and the things that accrue to that
Singular vision of what we want to build you should expect us to pursue
Got it
Do you think we'll be coding with command lines and agents and
In a few years or it's going to be completely changed?
I mean, I think that we're in a very unnatural state right now for how we work
Like we all sit behind this box and kind of type away and it's very clear our bodies were not designed for this
We got our carpal tunnel and our you know hunched shoulders all these things
And I don't think we want that. I don't think any of us wanted that like I think that we want more free time
But it's not even about free time necessarily, right?
It's like you want to spend more time with your loved ones
Yes, you want to spend more time like talking to people and like coming up with like
Brilliant visions or just like what you're excited about or just understanding yourself
So it's kind of like do you want to be a ceo of an organization of like a hundred thousand agents like that actually seems pretty good?
And I think that we're all going to be able to get so much more done
But the mechanics of it are going to feel as different as like going from having to
Write out things with you know by hand with a quill or something to
Being able to you know just send a text message and have people go in and you know working on your behalf on your goals
All right, we talked about compute. We talked about model and security and agents and app layer
Let's talk about frontier
When when are the models going to be good enough to push the frontiers of science physical ai?
It seems like we had gen found here. It seems like lms have been a great scaling law for digital intelligence
It hasn't been as strong for
robotics for physical intelligence for
aspects of
biology and science where the problems are probably a lot harder to verify or it takes a long time to verify
Well, how are you keeping track of science and physical and ai in in the world?
Well, science is one domain that we're really leaning into and we see
Line of sight to really incredible progress
We're starting to have some signs of life
And I think it's always important to ground in what is happening today when trying to predict what will happen six months a year from now
so for example, we had a physics result
where
Our ai came up with this very beautiful formula that
Physicists who've been working on this for quite some time thought was totally impossible thought it was like maybe an unsolvable problem and
Like it's pretty significant, right?
It's like real serious physicists who
Who really do this as a step towards really being able to get to?
To some sort of answer for quantum gravity and all these things not there
But it's a step that's much bigger than where we were just a couple months ago
And so it makes you really wonder a year from now
Like how far will we have traveled now things like biology that they are different from physics and math right that they are
You got to leave your beautiful simulated world and you know deal with messy reality
But I think we've been learning how to deal with messy reality in other domains software engineering is a perfect example
Where we've really realized that just building the thing that solves competition
You know programming competitions like that's not enough like you need something that's seen real world messy code bases
humans interrupting in different ways like this adversarial banging at it
And so I think that that on science I expect we're gonna see a real renaissance
You know, maybe we'll see some big results this year next year. I think it's going to be a totally wild wild time
We live in interesting times
I I promise that I get you out on time because you're a busy man before we let you leave
We got one minute on the shot clock what since you have no time but soon you will have lots of time
What do you and Anna do for fun fun?
Oh fun I mean
Same as anyone like like to watch movies go on hikes that those kinds of things
You know not as much time for it as maybe we'll hopefully have post-agi
But you got to kind of enjoy the ride along the way
Thank you greg for joining us. Thank you everyone