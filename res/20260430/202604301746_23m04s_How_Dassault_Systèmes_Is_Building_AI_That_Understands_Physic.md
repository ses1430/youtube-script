---
title: How Dassault Systèmes Is Building AI That Understands Physics | NVIDIA AI Podcast Ep. 296
uploader: NVIDIA
channel: NVIDIA
channel_url: https://www.youtube.com/channel/UCHuiy8bXnmK5nisYHUd1J5g
duration: 1384
upload_date: 20260429
webpage_url: https://www.youtube.com/watch?v=gsbi7FS8Q30
id: gsbi7FS8Q30
categories:
  - Science & Technology
tags:
  - NVIDIA
---

# How Dassault Systèmes Is Building AI That Understands Physics | NVIDIA AI Podcast Ep. 296

The agents can use the virtual twin as a gym to train themselves.
So they can run, in fact, millions of simulation or design experimentation
and present to you, to the human, to the engineers, the proven solution.
Welcome to the NVIDIA AI podcast. I'm Noah Kravitz.
My guest is Nicolas Cerisier.
Nicolas is vice president of the 3D experience platform R&D
for Tissot Systems.
We're here to talk about the next generation of agentic AI systems,
including industry world models, virtual companions,
and the systems that are driving them.
Nicolas, welcome to the NVIDIA AI podcast.
Thank you so much for taking the time to join us.
Thank you, Noah, and thank you for the invitation
and this opportunity to be part of this podcast.
Absolutely. The pleasure is ours.
So maybe we can start with you telling the audience
a little bit about Tissot Systems.
I have a long-running partnership with NVIDIA.
So you can speak to that a little
and then also to what your role is
and what the 3D experience platform is.
Okay.
So I'm Nicolas Cerisier.
I joined Tissot Systems in 2004
and I'm now the vice president
of 3D experience platform research and development.
And you have to know that the 3D experience platform
is really the foundation for our 12 brands
at Tissot Systems.
You know, I think the main brands, Catia, SolidWorks, Simulia, etc.
And if you don't know us,
we enable our customers to imagine, design, simulate, build
almost everything in the world.
Cars, airplanes, autonomous robots,
furnitures, electronic device,
therapeutics, med devices, etc.
It's 400,000 customers,
45 million users,
15 million scientists and engineers
all around the world
using our solution every day.
And in fact, we provide our customers
the factories to create their Virtual Twins.
And what is Virtual Twins?
It's really the scientific, multidisciplinary,
multiscale, V plus A,
virtual plus real representation
of the product you want to deliver.
And in fact, we enable a product
to be tested in the virtual world
in the real condition
before anything exists in the real world.
And so today, my focus
leading the 3D Express platform
is really to transform
our platform architecture
into an agentic platform.
And in fact, this is our shift
from a SaaS platform,
SaaS architecture,
to an agent-as-a-service platform
to bring AI to all our customers.
So much has happened
in the world of AI
in the past few years
and generative AI
obviously has been,
you know, this touch point
that set off large language models
and reasoning
and now we're talking
about agentic systems.
So let's talk about these two terms,
virtual companions
and industry world models.
And what do those mean
to Dassault in the Dassault world?
How do you use them?
And how are they different
from the types of generative AI
that people might be used to using
for the past few years?
Yeah, so let's start
with industrial world model.
Okay.
Our ambition, in fact,
is to build AI for industry.
It's very, very,
really important for us.
It's industries.
It's at the core
of everything we do.
And for us,
AI for industry
rely on three core principles.
It should be grounded in science
and this is what we do
for more than 40 years now.
We are a scientific company.
We deliver modeling technologies,
simulation technologies.
Then it should be fueled
by industry knowledge
and it should be sovereign
by design
from the underlying infrastructure
up to the models themselves.
So how is it different
from a generative AI?
I think a classic generative AI
learns the dynamics of the world
from the observation
and the perception of the world.
So let's imagine
they can see a video of a plane.
Okay.
They can predict
if the plane will take off,
if it will fly.
But in fact,
they don't really know why
because they don't have
the scientific explanation
and the scientific foundation
to understand that.
And obviously,
a plane does not fly by accident.
So in fact,
our industry world model principles,
they understand
how things work.
They really understand
the scientific foundation.
They include
the scientific,
physical laws of the world.
The physics,
engineering rules,
chemistry,
material science,
et cetera.
And they combine
the multi-scale,
multi-discipline modeling
and simulation technologies
we provide
with AI.
And the technology
we are delivering,
our industry world models,
rely on three technical pillars.
First,
the industrial knowledge.
Here we are talking
about the standards,
the regulations,
the processes
from the different industries
we serve.
And we embed
the real world
engineering rules
so the AI will understand
and will speak
the language of the industry,
the jargon of the industry.
Right, right.
You see?
Then,
the virtual,
the world understanding,
the world industrial understanding.
Here,
we are delivering
an ecosystem
of specialized
industrial AI models
which operates
on our virtual twins.
So,
the virtual
and real representation
of the product
you deliver.
Right, right.
and this integrates
the structure
and the physics behavior.
So,
combined with our
data system
modeling
and simulation
technologies
and solvers,
this is how
we can ensure
that the AI
will be grounded
in science.
And last
is the industrial
reasoning
and generation.
And this is where
the agentic
choreography
takes place
and activating
the industrial knowledge
and the world
representation
to perform
the experience
based reasoning.
And so,
about virtual
companion now,
in fact,
if the industry
world model
provides the intelligence,
the virtual companion
turns that intelligence
into action.
What we mean
with virtual companion
is we deliver
virtual companion
are your co-workers.
They understand
your intent,
of course,
but they will
resonate
with industry
world models
to orchestrate
execute action
in context
of your business,
of your industry.
So,
they will
comply
with the regulation,
with your KPIs,
etc.
Sure.
And they will
protect
your most precious IP,
of course.
And something important,
we don't want
to replace people.
We want
to augment people.
We want
to free time
to people
to innovate
and solve
problems.
So,
a few months ago,
we introduced
three virtual companions.
Pora,
the business expert,
Leo,
the engineer,
who solve
complex engineering
challenges,
and Marie,
the scientist,
who bring
deep scientific
expertise.
So,
when you're
designing
and deploying
the virtual companions,
and if we think
about
sort of
a workforce,
a virtual workforce
of companions
that,
as you said,
aren't replacing
human workers,
but working
side-by-side
with us.
In an environment,
like in a manufacturing
environment,
or industrial
environment,
where,
you know,
I think of my work
in content,
creating content,
podcasting,
and writing,
and if an LLM
hallucinates,
then,
you know,
hopefully I catch it
and I can make
the correction,
or maybe it inspires
me to something.
If a system
hallucinates
in an industrial
environment,
you know,
the consequences
could be much
more dire.
So how do you
build trust
into these systems
so that
the people
who are designing
and deploying
and working
in these environments
feel confident
working alongside
the virtual companions?
In fact,
I think the
foundation for trust
in our system
is the scientific
foundation,
scientific background,
then the
human in the loop,
because at the end,
human is accountable
and remains in the loop,
and the choreographie
will pause
when humans
have to take
decision
at the critical
milestone
of the execution.
And something
very important
we deliver,
and I think
which is unique,
is what we call
IPLM,
IP Lifecycle Management,
where we enforce
the lineage,
auditability,
traceability
of all
the interaction
of AI.
So we are
able to know
that your content
has been modified
through which
workflow,
using which
what kind of
models,
et cetera,
et cetera.
And we provide,
so we provide,
we provide
the source
of trust
to understand
how your virtual
companion
behave
with your content.
So NVIDIA
is bringing
technologies,
open models,
Omniverse,
accelerated computing,
AI physics libraries,
all these technologies
into the stack.
How do technologies
like these
help enable
more capable
and more secure
agentic workflows?
Yeah.
So NVIDIA technologies
in fact infuse
in every layer
of our architecture,
from NVIDIA AI,
with AI factories
for GPUs
and computing
infrastructure
to NVIDIA AI,
CUDAX libraries,
Omniverse technologies
to accelerate
AI training,
inference,
and simulation.
Regarding NVIDIA AI
and agentic,
we focus
on our partnership
with NVIDIA
on three axes,
understanding,
reasoning,
and execution.
Understanding,
we integrate
NVIDIA NIMS
models
into our
outscale
Kubernetes platform.
Outscale
is our
IIS.
It's a brand
from Dassault
system.
And we are
huge fan
on NIMS
because it's
super easy
to deploy
and
perfect.
Always glad
to hear it.
All our team
are in love
with NIMS.
Awesome.
Love to hear it.
So we leverage
NVIDIA open
models
for multimodality,
Riva,
Parse,
VLM.
And with Parse,
we improve,
for example,
by 30%
our document injection
and throughput.
Plus also
some industry-specific
models,
such as
Bionemo
for our
virtual
companion
Marie,
the scientist.
About
reasoning now,
we leverage
Nemo
3 Super
and the
reasoning
performance
for Aura,
Leo,
and Marie
have been
improved
by 20%
without
specific
optimization.
And this
is thanks
to the
collaboration
with NVIDIA.
We shared
our
industrial
use case
and benchmark.
And so we
were able
to iterate
together
and to
optimize
the model
and the
integration.
And then
about
execution.
With NVIDIA,
we are continuously
improving
the agentic
execution,
leveraging
the recent
announcement
of AIQ
Blueprint
and Deep Agent.
And we
are also
interesting
and prototyping
the recent
announcement
of Nemo
Claw,
of course.
And we
are exploring
Dynamo
to optimize
the GPU
utilization
and Nemo
agent
toolkits
for the
optimization
of our
agentic
workflows.
Can you
speak a little
bit to
the partnership?
You've mentioned
it as you've
been talking,
but just
kind of,
you know,
how it got
started
and more
kind of
what it
means to
Dissot
and what
it enables
you to do.
In fact,
for over
25 years
now,
as you said,
the system
and NVIDIA
have redefined
what is possible
together.
moving from
accelerating
pixels
to accelerating
computing
and now
to accelerating
industrial AI.
And so
back in
20,
back in 2000,
from acceleration
of visualization
of Katia
V5,
our flagship
brand and
app,
leveraging
NVIDIA GPUs,
to accelerating
computing
for Simulia,
Abacus,
and Xflow,
our simulation
brand,
with CUDA
and of course
GPUs,
to accelerating
and optimization
rendering
with IRA,
RTX,
and now
with DLSS.
And so
this year,
we are opening
a new chapter
in this story
with AI
and combining
NVIDIA technologies
within our
3D Express platform
to deliver
industrial AI
platform
to our customers.
I want to ask
you about
open and
proprietary models
and running
a hybrid model.
And my understanding
is that
Dassault runs
hybrid models
quite a bit.
Can you speak
a little bit
to kind of
the pros
and cons
of each
and why
you go
with the hybrid
models so often?
Yeah.
So,
yeah,
you're right.
We have
a hybrid approach.
Of course,
we build
our own models.
Yes.
But we want
to rely
on the best-in-class
frontier model
provided by NVIDIA
such as
the Nemo 30,
of course,
our optimized model
by NVIDIA
and available
through NIMS.
which,
as I said
before,
enable a seamless
deployment.
It's super easy.
Or we have
also a partnership
with other
model providers
such as
Mistral.
In fact,
we select
our models
and our partners
based on
the performance
of the model,
of course,
but also
about the
sovereignty
and the
regulation
constraint.
Okay.
Because we
operate
worldwide,
we have
a customer
in all
industries
and many
customers
and regulated
are very
sensitive
industries.
Sure.
So we
have to
comply
with our
own
regulation
and
all
the
auditability
problematic.
Right.
And so
from that,
we also
want to
calibrate
the model
with the
customer
knowledge.
So we
inject
the industry
knowledge
through
fine-tuning
or
RAG
depending
of the
use case.
Sure.
But more
generally,
we believe
in open
standards
and so
we embrace
and we
support
open
standards
such as
MCP
or
agent-to-agent.
In fact,
it empowers
our
agent-to-agent
platform
to leverage
third-party
industrial
system
and enable
in fact
interoperable
or
cross-system
agentic
choreographies.
I want to
ask if we
can dig in
a little bit
to a specific
use case
to kind of
get a flavor
for some
of the things
your customers
are doing.
Maybe if
there's an
example
that comes
to mind
you could
speak to
that really
illustrates
the use
of the
virtual
companions
and the
DeSoe
platform.
I think
one
super cool
example
I think
is
Leo
Mechanical
Designer.
Okay.
We showcase
this live
this new
virtual
companion
in our
3D
Express World
Conference
last February
with Jensen
attending
to this
conference.
And so
here
you give
Leo
a 3D
scan
or 2D
drawing
or a mesh
of a part.
He will
activate
the industry
world model
for design
orchestrate
the AI
model
and
the modeling
and simulation
solvers
and he will
perform a
multi-tier
planning
enabling
evaluating in fact
the mechanical
interface
of the part
finds the
physics,
the kinematics
and the
design rules
and at the end
it will generate
the optimized
design
physically aware
manufacturable
manufacturer ready
and it will
do it
right
the first
time.
It's a
very
super
example.
I think
it really
illustrates
our
transformation
from a
SaaS
to an
agent
as a
service
platform.
in fact
with that
we are giving
to our
millions
of designers
the power
to innovate
faster
but it's
not just
about speed
it's about
reliability
and trust
and because
you know
that your
design works
because it
is born
from science
from physics
and it's
augmented
with your
industry
knowledge.
that change
that you
referenced
from a
SaaS
company
to an
agent
as a
service
company
kind of
from a
philosophical
standpoint
I guess
or an
emotional
standpoint
does it
feel natural
is it
a big
shift
is it
just
kind
of
part
of
you know
the way
of doing
things
to keep
innovating
and delivering
for your
customers
and so
it's
just
kind
of
the
natural
progression
of
things
how do
you
think
about
it
in fact
with the
rise of
AI
we think
ourselves
what is
the deep
impact
of AI
in what
we do
and what
we deliver
what will
be the
new
experience
for the
user
what will
be the
new
technology
what
we all
see
the
cloud
code
etc
what if
you apply
such
transformation
to our
industrial
software
in fact
so it
came
from that
in fact
really
and so
this is
a lot
of discussion
and
brainstorming
at the
system
and in fact
we don't
want to
add
AI
on top
of
what
we do
we want
to put
AI
at the
core
and this
is why
we are
working
with NVIDIA
on the
different
topics
what's
a typical
way
to get
started
what's
a first
project
that
a customer
might
typically
undertake
to get
started
with
virtual
companions
and working
with them
I think
you should
start from
your core
business
and your
core
challenge
in fact
of course
this is where
you will have
attention
from your
teams
this is where
you have
your knowledge
your deep
knowledge
and your
deep
know-how
and this
is where
this is
how you know
to measure
the real
impact
of your
AI
and
agentic
transformation
right
and we have
an example
connecting to
LEO
mechanical
design
we are
working
with
NAYAR
and
NAYAR
is one
of our
customers
working
with us
on
Virtual
Companion
and
what
they are
doing
to do
is
they
recreate
the
virtual
twin
of
existing
aircraft
it
means
that
they
are
creating
thousands
of
parts
without
access
to
the
original
design
so
basically
they
disassemble
the
aircraft
and
recreate
virtually
piece
by piece
right
wow
so
of course
with LEO
you can imagine
how it
changed
their life
automatically
generating
the 3D parts
from their
multiple
sources
that's
incredible
so
like
everything
else
in technology
in AI
now
virtual
twins
virtual
companions
simulation
just
accelerating
advancing
so
quickly
and
obviously
agentic
frameworks
and
models
are
developing
just as
quickly
if not
faster
what's
next
what's
on the
horizon
for
Dassault
systems
what are
the kinds
of things
you're
thinking
about
and
then
if you're
game
to take
it a
step
further
where
do
you
think
agentic
systems
and
the
idea
of
virtual
co-workers
is
headed
okay
first
I think
the
system
strategy
is
fully
aligned
with
the
recent
NVIDIA
announcement
about
Nemo
CLO
AIQ
all
the
agentic
stuff
and
the
rise
in fact
of
the
long
running
autonomous
agents
and
we
fully
agree
on
the
associated
industrial
challenges
security
compliance
etc
and
tomorrow
our
virtual
companion
Aura,
Leo and
Marie
we
believe
they
will
stay
awake
and
they
will
continuously
monitor
your
factory
your
project
execution
your
supply
chain
in real
time
and
they
will
proactively
optimize
it
optimize
the
virtual
twin
without
being
prompt
by
human
so
it
will
create
in
fact
I
think
a
closed
loop
autonomy
and
because
of
our
industry
world
models
are
grounded
in
physics
I
think
the
agents
can
use
the
virtual
twin
as
a
gym
to
train
themselves
so
they
can
run
in
fact
millions
of
simulation
or
design
experimentation
and
present
to
you
to
the
human
the
proven
solution
and
you
just
have
at
the
end
to
validate
and
from
that
the
virtual
twin
in
fact
become
a
self
evolving
asset
that
gets
smarter
day
after
day
in
fact
Nicolas
there's
so much
going
on
for
listeners
who
want
to
learn
more
about
the
3D
experience
platform
about
Dassault's
work
with
everything
we've
talked
about
virtual
companions
and
industry
world
models
where's
a good
place
to
go
the
Dassault
website
social
media
are there
research
papers
where can
listeners
go to
learn
more
mainly
on the
Dassault
system
website
3ds.com
or on our
LinkedIn
page
where we are
communicating more
and more
on AI
thanks also
to the
NVIDIA
collaboration
we are posting
more and more
about
what we are
doing
so yeah
perfect
that's free
and connect
with us
excellent
well
Nicolas
again
congratulations
on all
the work
and thank
you for
the years
of collaboration
with NVIDIA
thank you
and best
of luck
and everything
you're doing
thank you
to NVIDIA
to the team
incredible team