---
title: Harrison Chase of LangChain on Deep Agents, LangSmith, and Earning Trust | NVIDIA AI Podcast Ep. 297
uploader: NVIDIA
channel: NVIDIA
channel_url: https://www.youtube.com/channel/UCHuiy8bXnmK5nisYHUd1J5g
duration: 1495
upload_date: 20260506
webpage_url: https://www.youtube.com/watch?v=c-fsL0gsmo0
id: c-fsL0gsmo0
categories:
  - Science & Technology
tags:
  - NVIDIA
---

# Harrison Chase of LangChain on Deep Agents, LangSmith, and Earning Trust | NVIDIA AI Podcast Ep. 297

and so i think like these always-on asynchronous event-driven agents that will be a really big
productivity unlock and especially in enterprises there's so many events that are just triggering
triggering triggering and so if you can have agents listening to those and firing off i think
that will be a massive game welcome to the nvidia ai podcast i'm noah kravitz our guest today is
harrison chase harrison is ceo and co-founder of lang chain um one of just the most incredible
stories of this whole generative ai era that we're in as harrison will get into in a minute lang chain
was founded about three years ago over a billion downloads uh the whole point is to help developers
build applications with llms and now getting into agents and agentic frameworks and all that great
stuff so we're going to get into it in a moment harrison thanks so much for joining the ai podcast
and welcome thanks for having me excited to be here so let's start about three years ago um you started
laying chain with this premise of building tools so developers could build apps with llms
um what did you see back then that either others didn't or even if they did you saw and just thought
this is where things are going this is where where i'm headed what got us really interested was seeing
the applications that people were building on top of llms right and the systems they were building around
the llms in order to power those applications and those systems had a lot of similarity with each other
even early on and even early on we could tell that they would get quite complex over time and so a lot
of what we built is tools to help people build these systems these agents which we now call agents around
around these llms and and figuring out what the common patterns are and the condoms tooling is and
making it really easy for anyone to do so right um and so now you've coined this i don't want to say this
term but i lang chen you talk about uh deep agents yeah so what is a deep agent and then maybe we can get
into talking about the enterprise and why would an enterprise in particular care about that distinction
yeah so about a year ago we saw a few really interesting things so maybe even backing up like
you know three years ago great like llms you want to connect data you want to connect these other things to
them fantastic how do you do that turns out it's really hard and the best way to do that for for
different types of agents was actually pretty different you would build different scaffolding
you would build different workflows around the llms about a year ago we saw cloud code come out we saw
manis come out we saw deep research come out and under the hood all of these had the same kind of like
general architecture they were they were simple in some ways they were an llm running in a loop calling tools
but then they also had common patterns of like connecting to a file system and having sub agents
and doing planning and and and and so about like nine months ago we released uh for the first time deep
agents which was which is the library and we've been building it ever since and we've just continued
to see kind of like this same pattern of of of giving the llm more autonomy in this environment for
interacting this is what powers open claw for example is this this type of harness and so deep agents is
really this new type of agent harness that we think is really general purpose and that you can customize
to do different things but it's not like you're reinventing the scaffolding each time you're just
customizing it with prompts or tools and so it's way easier to get started with and also way more
powerful because it's a simple thing under the hood and simple is is really good and so and so deep
agents is this general purpose agent harness model agnostic open source that we've been building for a while
and and we're starting to see more and more agents build on top of so when you're when you're working
with with customers and you know the enterprise in particular and we're getting into these systems that
are so powerful becoming so powerful in large part because they are autonomous to a larger degree and as
you said they can do more now the agents can control the screen and go off and do things with apps and such
what's the what are your conversations like with enterprise leaders and what's kind of the the
feeling around you know is it a tension between risk reward is it just the excitement for what the
systems can do and so there's trust in building these systems that give agents more leeway what were
those conversations like there's a lot of things so one like you know not not everything needs an
autonomous agent sure and so one framework we have lane graph is really good for when you actually want
to combine some of the autonomy of llms with more more directed workflows and more control and and so
honestly when talking with a lot of enterprises about deep agents some of them are just like we love
lane graph lane graph is better we're going to stick with lane graph sure and that's fine with us we
think there are different use cases and different things but that that's definitely kind of like one
component that comes into it um another component that comes into it is definitely just like okay great like
the lm's doing a bunch but how do we know what's going on sure um and so another thing that we work
on is laying smith which is observability and evals and that's basically our answer for that you you
there's this really interesting thing about agents compared to software where we're agents the the
the interaction space for agents is is way more open-ended you can ask it anything like text is infinite
if you have a ui there's a bunch of different buttons you can click and so it's much more constrained
and then also models are are not robust at all like you know they're non-deterministic and then you
change one word and the answer changes completely so this is why we think observability is really
important and that's a huge thing that enterprises care about and very related to observabilities than
evals because sure you can you can see one thing that happens you can tell why it goes wrong but what
if you know what if you want to test how it did on like 10 different questions 100 different questions
and so building up these eval data sets is a big thing we work with folks right and so langsmith is the
platform for building agents as well as observing and evaluating yeah so the way that we think about the
the the agent development life cycle is build test run manage okay and so the build is all the open
source you can it's like choose your fighter choose line graph choose deep agents choose another framework
all of our stuff works kind of modulary but then this test run manage that's langsmith so we've got a
bunch of stuff around testing and evaluating these models we have a deployment platform for deploying these
at scale and then we have observability and other things for managing them let's talk about skill or maybe you can talk about skills for a minute when i first started
playing with these tools and i'm you know i'm not a developer i'm just kind of a technical layperson
if you all right i love playing with these things back in the day when they first came out uh maybe it was baby agi
whatever it was i you know spun my computer right into the ground in an infinite loop um but when i first discovered
skills it took me like there was a moment where it sort of took me back where i was like wait i just describe
it and it goes off and it built and then of course it does because that's how this all works but can you
talk a little bit about skills and about kind of that that same idea of giving the agent the autonomy to
write the tool and run with it but how do you you know keep it in check and keep things secure yeah skills are a great
skills are a great way to package up knowledge and in other kind of like instruction sets and other tools
for an agent to use and so they started in coding agents and a skill would involve basically a markdown
file with some instructions and then some scripts that you could run and and one of the things that's
kind of become clear over the past few months is like coding agents are very general purpose in a lot of ways
and so this same idea of a skill as a markdown file and then some scripts to run is is really really
interesting um we see a bunch of different types of skills some of the skills are purely kind of like
informational so like you want to learn about something great go go read this markdown file
other skills do things and this is where it starts to get i think like really interesting it
could be a python script that that hits a url it could be a python script that runs some gpu
accelerated compute um and so this is this also ties into the environment aspect so so when we think about
agents we think of a model a harness and this is deep agents and then an environment that it runs in a
runtime for it right and so nvidia just released open shell which is a secure runtime for it and then the
other thing that's related to the runtime is also like where it runs does it run on a mac mini does it
run on on some gpu accelerated environment does it run in the cloud right and so those three components
and being able to pick and choose what you need for for different jobs is is a big part of kind of like
customizing your agents was there a moment or can i put you on the spot and ask you to think of kind
of an aha moment where this the idea of deep agents really clicked and and in a use case and whether it
was you know something internal at langchain you're working on or maybe with a customer was there kind
of an aha moment where you were like yeah this is this is it i think so it started just by seeing
really the three things of manis deep research and cloud code and this is the same way that langchain
started as well just going to early meetups seeing things that people were building and seeing patterns
right and and so the first version of deep agents um just like the first version of langchain uh i
hacked on over a weekend um and it was a weekend project i'd been talking internally with some folks
and being like oh like you know cloud code's really interesting like manis that's they've got some
similarities and so it wasn't until i had time to kind of like sit down on a weekend and and hack some
stuff together that it you know we we came up with a few patterns of what these similarities actually
were um and then and then using it i think the first thing we used it for was a deep research type
thing um and so we gave it we gave it access to a bunch of files and we just put it in this like
virtual file system and had to do some research and and it wasn't even really doing rag it was just
grepping and globbing like like a coding agent would over these files and it worked fantastically well and
so i'd say deep research was the first concrete thing but really the idea came came from just seeing
seeing a pattern and spending a weekend kind of like hacking on it you um you mentioned earlier uh
the importance of you know with the words you use but auditability traceability being able to see
how did the agents you know do what they did um can you talk a little bit about evaluation driven
development yeah and how that plays into and again in the enterprise you know building that trust in what
the agents are doing yeah if you talk about trust in an enterprise that that you know what does that
mean that means that the agent's doing what you want it to do there's a few different ways that we
see people getting that trust part of it is observability and traceability and being able to go
into an agent run and see exactly what steps it took and exactly what it did the other part where trust
comes in is is having like this these these scenarios and running the agent over them and seeing how it
performs and evaluating that and and this is kind of like what we talk about as evaluation driven
development you come up with these scenarios ahead of time one one common misconception here by the
way is that you need like a thousand scenarios for it to be effective you could start with five you
could start with ten it really doesn't matter i i think like creating these evals is a really good
way to do like product thinking about what the agent should act because this is another thing like
agents can do anything yeah but they shouldn't do everything they should do like what you want them to do
and so being able to be being forced to come up with like hey these are 10 questions that we expect the agent to get asked
this is what we think a good response is for each of them this is what a bad response is for each of
them that's a really good kind of like mental model for kind of like coming up with what these agents
should do and then you can use that to drive all of your changes so you change a prompt great you
can run it against this benchmark did it improve did it did it did it get worse and then this this eval
data set is is living over time as well so as you release it to first like a small set of users you
might see them using it in unexpected ways and and and then some of those ways you might be like okay
maybe they shouldn't be doing that let's put some guardrails around it but other ways you might be
like yeah that's totally legitimate we had no idea they would use it let's add some data points to our
eval data set so when we go and change the prompt in the future we can make sure that it's still good
at these use cases are enterprise customers open to kind of rolling with that you know oh we weren't
expecting this behavior necessarily but it's good behavior and so you know is it is there a sense of
kind of experimentation obviously in the ai community and the open source community it's all about
experimenting and sharing and things are going so fast is the enterprise embracing that at all the the
best ones do and in in limited ways and with a limited blast radius they might roll out internally for
example they might roll it out to a set of like alpha customers they might roll it out to one percent
of users or something like that right um there's definitely way more caution there than there is
with gen ai native startups but but building agents is so iterative um and in the importance of this
iteration can can really not be understated and so i think the the enterprises that are i i think a a
failure mode for enterprises is you have some idea of an agent you take three months to craft a bunch of
examples you take another three months to to build the agent you take another three months to get humans to
look at everything right but the space has just moved so fast like the the whole the whole idea you came up with there's probably a
better like there's just a better way to do it at that point and so i think like you have to
kind of ship you have to learn you have this is another thing by the way that no one likes the answer for but like
you have to you have to basically redo your agent every every nine months at the pace that
things have been like with these agent harnesses if if you're using an agent
architecture from like a year and a half ago you should very strongly be considering looking at
rewriting on top of an agent harness or something like that right and again we're for performance
only or for yeah yeah for performance that that's still the bit like um that so there's two things
it's like performance but also scope of what the agent can do so if the agent's doing a very small
thing it's not as valuable as if it's doing a big thing and maybe like a year and a half ago you just
couldn't get it to do the big thing so you focused on the small thing but now you can and so if you're
not like re-evaluating that and saying hey there's this big thing let's hook up an agent harness let's
take a step at that you you absolutely need to be doing that so i want to ask you about models um
frontier models you know the uh i would say everybody but i think the kind of mainstream
ai world you know focused on the latest and greatest and what can they do and everything open models have
become incredibly important i mean they've always been important but i feel like the past year or so
incredibly important you know you spoke earlier about um open claw and nvidia's open shell and the
nematron family of models how does how do you approach and how's langsheng approach and then your
customers mixing frontier and open models together to achieve you know cost performance ratio and and all
manner of other things what's your uh what's your approach on mixing mixing those yeah i i think
there's a bunch of different ways that we combine them so i think like one one obvious way that we
worked with nvidia on a blueprint for is is with deep research you have a bunch of sub-agents and those
sub-agents might want to be specialized agents and there might be a there might be an orchestrator kind of
like agent that's using a frontier model but then when it goes to a sub-agent it might want to use either
like a fine-tuned model or an open source model for for costs or latency reasons and so when you have
these big agentic systems with these sub-agents it's totally possible that one part could be using a
frontier model and one part could be using an open source model and another part could be using a
fine-tuned model um the other we've been paying a lot more attention to open source in the past even
just like two weeks i would say for probably two reasons one um i think they're getting good enough to
where they can drive this harness so so you know the like being able to properly utilize everything in
the harnesses is not it's not super easy and for a while it was only the frontier models that could do
that we're starting to see there's still a step below the frontier but we're starting to see that
these open source models can drive the harness which is really interesting because this is the most agentic
stuff yeah and then the other thing that's causing us to look really hard at open models if i could stop
you for a second harrison back up what are the qualities that a model needs to drive the harnesses
successfully so at the risk of signing a little broad like it needs to be intelligent it needs to
be good another thing that that is maybe underappreciated is it it probably needs to be good at
coding okay so we've actually seen that like quen coder is a better general purpose model than just the
quen series of models interesting because a lot of what makes up this harness looks very similar to
coding agents so this harness has a file system it has a bash tool right so if the model knows how to use
it if it's a coding model then that's actually really really good and so i think models that are better at
coding are generally actually general better general purpose agents yeah no that makes sense and so then
the sub-agent models you're talking about yeah and so then a second thing that made us look at this
um look at open source models even more is is open claw so one of there's a bunch of really interesting
things about open claw but one of the interesting things is it's always on it's proactive it's running
and so if you know if you're using a coding agent and you kick it off even let's say like 20 times a
day you know you're probably okay paying like some good amount for that if it's running every 10 minutes
like oh oh my god you cannot and if you if you're running like three of these like you just cannot
do that and so i think like cost is a really interesting um reason for these open models especially
in these proactive always-on scenarios to to make them become popular um shifting gears for a second uh
langchain just opened nvidia formed the nematron coalition and langchain joined can you talk a little bit
about um why and what it may or may not mean going forward for langchain users yeah we we need open
models and we need harnesses that they can run in um and you know we we think we can provide the harness
and we want to work with nvidia and all the other companies in the coalition to to help provide a model
that can that can work with that harness and others as well um i think like you know as we talked about
like the open source models are getting good they they're still a little bit behind the frontier models
in terms of driving the harness and so great we can use them in sub-agents you know we can maybe use them
um uh for some of these kind of like triggers and the always-on but if they can drive the really
expensive workloads i think that's going to be really transformational in terms of um what you can do
with open models which generally mean what you can do with more sensitive data what you can do
uh more cheaply what you can offer to customers just more yeah um and so yeah i think at a really high
level we're excited about the nvidia uh nematron coalition because we want we want an open model
that works really well with open harnesses and then a third part which actually i don't think was part
of the the coalition when it started but i think the open runtime is really important as well and you
guys are also doing stuff around that um this is my favorite question to ask and you know i'm sure
the hardest but maybe the most fun to answer um what's next what do you think agents agentic systems
langsmith langchain the company for that matter is going to look like in and i'll let you kind of go
with what time frame makes the most sense because i ask and depending on the guests they're like a
year no that's too long no no no um but what do you think's coming down the pike as far as you know
agentic systems and and all of these things that you're working on every day i i'd maybe call out
kind of like three things that i think are interesting um one's one's pretty short term and i think we'll see
in the next like month or two if if not if not by the time this comes out but but uh asynchronous sub-agents so
right now when when an agent kicks off a sub-agent it basically waits for it to respond and and and
that's great but if these sub-agents start to get really long running you want to just have them run
in the background and you want to have this manager orchestrator agent like check in on them and maybe
update them and so i think one trend that we'll see is is encoding right now encoding agents you interact
with the agents that's doing coding i think we'll start to see a trend where you interact with this
orchestrator agent and that orchestrator agent spins up a bunch of background coding agents and you just
talk to the orchestrator and say hey what's going on with this experiment what's going on with this
feature right so i think we'll start to see asynchronous sub-agents become a bigger and bigger
topic i hate to resort productivity but how much of it is that going to be a step change or how much of a
difference in terms of what you're able to accomplish so so i think i think this bill like the only reason
asynchronous sub-agents even make sense is if the agent the sub-agents themselves actually run for a
while right like if they just run for like one second and return you can just make them synchronous
and and and so i think like it will be a productivity gain but it like requires these agents to be long
running in the first place and i think that's the real productivity gain and i think this is just a nice
interface on top of them what one thing um that i that wasn't on my list of three things but i think will
also be more and more impactful is basically these agents being proactive running in the background
always on listening to events that i think will be a massive productivity game so i have an email
agent it runs in the background it listens to my emails um when when it wants to respond they're
still human in the loop but it like it like flags a draft and it's like hey here's a draft do you want
to approve it do you want to change something that is so much more efficient than if i had to go
like there's no way i would take an email copy paste it go to chat gpt say hey can you draft me a
response copy paste that like that and so i think like these always on asynchronous event driven agents
that will be a really big productivity unlock and especially in enterprises there's so many events
that are just triggering triggering triggering and so if you can have agents listening to those and
firing off i think that will be a massive game the the other two things that i think are coming down
one agent memory we started to see this a little bit with open call but i think the idea that it could
remember things as you interact with it it could actually update its own tools and skills and
description itself i think more and more we'll see agents kind of like remembering things and and and
and um yeah learning from their interactions and that's why that's why human in the loop is important
as well that's why i don't think these things will be fully autonomous because they need to learn
and the only way you do that is by interacting with the environment with humans and so i think that'll be a
big piece of it and then the last thing is agent identity so um you know if there's an agent in
enterprise and i chat with it and you chat with it whose credentials does it use does it use mine does
it use yours does it use a fixed set so previous to open claw i think we saw that basically everyone
was doing the the on behalf of model right so the agent would act on behalf of me on behalf of you on
behalf of the end user and i would pass like my slack credentials through and so i might get a different
answer than you would get i think the thing that open claw changed is people started thinking of these
agents as like identities as their own as their own things and i think we'll actually see more things
where they will be like hey tom is a marketing agent and you can chat with tom and i can chat with
tom and tom has a persistent memory and tom has its own credentials and tom can go and do things and
tom is tom tom is not acting on behalf of me or you tom has its own accounts with with slack or gmail and
that's a big thing that we need to figure out that i don't think anyone in the industry really knows
you know i was chatting with one sas provider they they made they went in all the open claw craziness
they were making it really easy for people to create accounts for their agents but it's still like an
account and so like will we see will we just see more and more people create normal accounts will there
be special agent accounts i don't know but i think this idea of like agent identity is really interesting
yeah um there's a whole can of worms on the other side of the words agent identity i think but
not for this conversation um so you know you mentioned the weekend project um that that you
worked on that unlock things at lang chain for you uh open claw another weekend project went incredibly
viral incredibly quickly um what what are your thoughts or how has that impacted the work you do
and i'm thinking more about the perception that that users developers enterprise customers might have
about agents as it really you know has it was there an a rush of people knocking at your door saying
like hey can you build me a claw like how does it change things a hundred percent i mean i think jensen
said uh what do you say every enterprise needs a claw strategy or something like that and we're absolutely
seeing that i think like it's set a north star it's set a new new objective for for for kind of like
what these agents can and should be able to do now there are a lot of things that you probably want to do more
securely than then kind of like in an open claw the whole reason it took off is because it can it can do
everything and that's great for for weekend projects and hobbyists but when you bring it into an
enterprise you're understandably going to want more want more control that's why we're thinking about
agent identity that's why we're thinking about observability but in terms of like did it change
the north star for for what we build absolutely it did i think it it also made it really it made it so
much easier to communicate some of the ideas as well and so that's that's been fantastic as well amazing
harrison that so much we just talked about in a short amount of time and so much more but i'm sure
by the time we cross paths again you know as you mentioned right you get your take three months to
scope and three months to build and all of a sudden it's nine months and you know no more so the next
time we cross paths i'm sure it'll be a different looking world but kind of built on these same things
but for folks who've been listening or watching and want to learn more about lang chain the work you're
doing best places to go online website socials research blog anything like that yeah we have a
great blog it's blog.langchain.com a lot of the stuff we talked about around context engineering and
agent identity will be blogs on there and we update that a lot and then and then twitter i think all you
know everything in ai is happening on twitter uh we're just we're just laying chain on twitter and so you
can find us there easy enough harrison chase thank you so much been an absolute pleasure appreciate you
taking the time to join the podcast thank you for having me