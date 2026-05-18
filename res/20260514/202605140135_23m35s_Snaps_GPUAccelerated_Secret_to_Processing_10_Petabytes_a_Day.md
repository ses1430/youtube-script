---
title: Snap’s GPU-Accelerated Secret to Processing 10 Petabytes a Day | NVIDIA AI Podcast Ep. 298
uploader: NVIDIA
channel: NVIDIA
channel_url: https://www.youtube.com/channel/UCHuiy8bXnmK5nisYHUd1J5g
duration: 1415
upload_date: 20260513
webpage_url: https://www.youtube.com/watch?v=glT-zO8B_qk
id: glT-zO8B_qk
categories:
  - Science & Technology
tags:
  - NVIDIA
---

# Snap’s GPU-Accelerated Secret to Processing 10 Petabytes a Day | NVIDIA AI Podcast Ep. 298

We were able to cut almost about 76% of our job costs as a result of this migration.
76?
76. It's phenomenal.
I mean, for the engineers out there, we were able to cut down the number of cores required by like 62%.
The memory footprint, we could drop it by like 80%.
So phenomenal results. The results speak for themselves.
Welcome to the NVIDIA AI podcast. I'm Noah Kravitz.
I'm here with Prudvi Vatala.
Prudvi is the head of engineering platforms at Snap, and we're here to talk about data processing,
and in particular, how a social platform with more than 940 million active users accelerated their data pipeline.
Prudvi, welcome to the NVIDIA AI podcast. Thanks so much for taking the time to join us.
Yeah, thanks for having me here, Noah.
So maybe we can start with the basics.
Tell us a little bit about, well, about what Snap is now.
I'm old, but I still think of it, you know, the snap glasses and everything, but Snapchat, obviously, a huge social platform.
So maybe tell us a little bit about Snap and then your role there.
Absolutely. Yeah.
I mean, Snapchat at this point is pretty much a household name.
You know, Snap as a company, it's interesting that you bring up the spectacles because Snap as a company believes that camera is at the center of, you know, improving how people communicate and improve their lives, you know, in the digital world, so to speak.
So we've been steadfast on that belief, and, you know, Snap right now is at the intersection of augmented reality, AI, and visual communication, like you said, serving close to a billion monthly active users.
I've been at Snap for a while now, and I lead a multifaceted organization.
We do a little bit of it has to do with big data infrastructure, a little bit of it with developer productivity, and a little bit of it with enterprise AI and whatnot.
So, yeah.
And so when we talk about accelerating data processing, what does that mean to you?
What does that mean for Snap and thinking about the scale that you operate on?
Just talk a little bit about what it means to accelerate data at that level.
Absolutely.
That's a great question.
Like, as you can imagine, with as many users as we have, and Snap, Snapchat in particular, is a very complex application.
So you can imagine the scale at which we operate, especially on the data processing side.
We are dealing with my team's experimentation platform is dealing with 10 plus petabytes each day.
It's a massive scale, right?
It's a huge scale, yeah.
And then we have a strict SLA in the morning because experimentation results need to be ready for developers, product managers, data scientists to act on as early as possible so that, you know, they can take appropriate action.
So, for us, accelerating data processing basically means instead of throwing more and more CPUs at the problem, figuring out a way to flatten that scale curve, you know.
So, in this particular scenario, it was about figuring out how to leverage GPUs for improving our workloads, making sure they run faster, cheaper, and scale, you know, linearly or sublinearly, unlike, you know, right now.
It's definitely super linear with feature areas.
So that's what accelerating.
So you mentioned experimentations.
What does that mean?
What are, when you're conducting experiments at Snap, what does that look like?
And then maybe how does that fit into, is that where the 10 petabytes of data each morning comes from?
Or we can talk about that.
Yeah, absolutely.
So this 10 petabyte data is only about the experimentation platform.
The big data across Snap is far wider.
Sure.
So experimentation, it's a little bit about Snap's product philosophy.
Like, we are, we believe that experimentation, safety, and privacy are core pillars for our product development and iteration.
Like, when we are thinking about new product areas, when we are shipping new product features to our, you know, half a billion daily active users across the globe, we need to think about how the users are receiving it, how they're responding to it, how they're using it, whether or not this is adding value to their, you know, daily lives.
And also guard railing things, like, is it regressing their performance, you know, is it causing their devices to slow down or, you know, we need to be very particular about protecting their experiences as well.
And so, Prue, along those lines with the experimentation, can you talk a little bit about the importance of A-B testing?
So, A-B testing is, you know, the concept of randomized control trials has been around for a long time, you know, especially in the clinical fields and whatnot.
But with the digital revolution, it has become the mode of bringing statistical rigor to decision-making at scale, right?
So, that's what A-B testing adds to us.
Like, you know, when we are dealing with this massive user base that is diverse by nature, you know, from all walks of life across the globe, you know, and we are trying to delight them, we are trying to bring experiences to them.
We need to make sure, you know, what we are delivering is buttoned down.
Like, it's actually really adding value the way we think it is, right?
And at this scale, a lot of things, you know, can happen.
And that's where having the statistical rigor grounded in, you know, holdouts and, you know, well-defined controls and statistical methods comes in.
Like, over the years, my team has added a bunch of statistical methods to our platform, you know, heterogeneous treatment effects detection.
For example, you know, you may think that a feature is performing well for the global audience, but it may not perform so well for a subset.
Right.
So, figuring out those heterogeneous effects is one thing that we focus on.
And, you know, at this scale, no matter how you slice your experiments, you're still allowing some bias to seep in, as in, you know, some power users may end up on one side of the experiment rather than the other.
So, how do we make sure the distributions are evened out when the experiment results are red?
That's the variance reduction aspect.
So, that's something my team built over time.
And then, you know, sometimes when we ship a feature, if people don't like it, they might even just stop showing up, you know?
Right, right, right, right.
So, that's the sample size mismatch problem.
So, we also do a bunch of that rigorously.
So, that's what A-B testing brings to the table.
So, with all of the data processing every day, what made you think that maybe some NVIDIA tech put into the stack might help things out?
How did that process start?
And maybe you can talk about, you know, what you've integrated and what you're using.
Absolutely.
So, I'm really proud of this.
I'm really proud of my team because over the years that I've been seeing our platform, the number of users grew, like Snap, you know, ballooned, right, in terms of footprint.
The number of features we shipped, like, you know, Spotlight, you know, AR features, AR lenses, and all of the AI features we shipped in the recent past.
So, they've also been adding a lot of additional dimensions to the platform.
And my team was hard at work making sure we are not, you know, we're scaling appropriately, even as all of this scale grows.
And they've done a very good job of it historically for years now.
Maintaining the cost flat and, you know, performance predictable, meeting the SLAs and whatnot.
And one thing we came across, you know, we came across NVIDIA Spark Rapids on one of the blog posts.
And we saw NVIDIA is shipping this, you know, solution to speed up our PySpark workloads by anywhere from 3.6x performance versus 50%, you know, runtime, you know.
It was phenomenal on paper, right.
Yeah.
So, that's what drew us to it.
You know, I'm waiting to hear them.
The numbers sound good.
I'm waiting to hear the rest.
Yeah.
So, we read those and we got super excited.
And then we, our stack was, it still is entirely Google Cloud for experimentation platform.
We loved working with them.
The Google Cloud data proc was phenomenal.
They've been a fantastic partner to us throughout the scaling journey.
So, when this, yeah, and then when this news came out with Spark Rapids, we wanted to try it out.
We did a bunch of benchmarking.
We tried, obviously, or like I said, we do a lot of things.
So, there is a lot of complexity to the nature of the jobs we run.
So, we had to benchmark each kind of job as well.
Right.
Like, you know, taking jobs that are heavy with joins and repartitions and, you know, shuffling
of data that moves data around versus, you know, jobs that are purely unioning data from various
places versus, you know, jobs that are purely aggregating, like running sums and whatnot.
So, we had to benchmark across all of them and we noticed that even on Google data proc with
Spark Rapids, we got about, you know, I want to say 3x plus, you know, improvement for the, you
know, join jobs and about close to 2x for, you know, the union jobs and a little over 1.5x for
aggregations.
That's largely because CPUs are already good at aggregation.
Right, right, right.
So, and then the other thing is GPUs by nature support parallelism and high bandwidth memory
on the hardware itself.
So, that made it like a very good candidate for us to pursue.
And so, you're running your GPU accelerated pipelines on Google Kubernetes, is that right?
Yes, yes.
That has been a very interesting journey from, you know, testing out our pipelines with data
proc, uh, GPUs and, and to today.
And, and one other thing, like with Spark Rapids, I want to mention it.
Uh, we didn't have to change a single thing about how we ran the jobs.
So, that was the beauty of it.
Not at all.
Zero code changes.
Oh, that's amazing.
Zero code changes.
Yeah.
So, um, I'm, I'm into developer productivity and developer enablement.
So, for me, that was music to my ears.
Sure.
Of course.
It was really impressive.
So, with data proc, uh, which abstracts out the Spark runtime for us and Spark Rapids, which
didn't require us to change the jobs, it was phenomenal.
Yeah.
So, it, it, it went very well.
So, we wanted to productionize this.
Um, we, we were able to, um, at our scale pipelines aren't just monolithic, right?
We, we do a bunch of sharding and then, you know, batching of work.
Uh, so we were able to migrate one shard to production on Google data proc using 300 GPUs.
Uh, the results were phenomenal.
Yeah.
And then in the next phase, we, we wanted to migrate 10 shards for total, you know, 50 plus
shard architecture.
And then it, it needed about 3000 GPUs, which was still doable with, uh, data proc on-demand
GPUs.
Okay.
Because, you know, GPU capacity is on everybody's minds these days, right?
Yeah.
That was well and good, but then, uh, we didn't have a path forward after that, right?
Uh, so we, we kind of, uh, hit a roadblock with, uh, you know, on-demand GPU capacity.
So we had to get creative.
So we started looking around, we were like, where at Snap do we have GPU capacity that we
can borrow?
Right?
And, uh, you know, uh, that, that's where the real insight came for us.
Like, Snap has a global audience and, uh, the Snapchatter's behavior is cyclical during
the day, right?
People wake up, they use Snapchat and they go to bed, they don't, right?
So what that meant was when some of our biggest markets went to bed, a lot of our online inference
GPU capacity was sitting idle.
Yeah.
Somewhere between 1:00 AM and 5:00 AM.
You know?
Yeah.
So that was, that, that was our opening, our opportunity to go tackle, uh, and that brought
about its own set of complexity, right?
Because online serving stack is not built for batch data process.
Okay.
Yeah.
Right.
They're, they're fundamentally, they were considered fundamentally different words.
Right?
So all the online, um, GPUs were tied to Kubernetes and GKE and we were already on Google Clouds.
And GKE wasn't, um, an issue for us at all.
It was actually very welcome.
Um, so we had to migrate our workloads to Kubernetes based Spark runtime and, uh, host it on GKE so
that we can leverage, you know, what, what the online GPUs had to offer.
And, uh, for that, we had to actually build a data platform ground up.
Okay.
You know, because, uh, uh, it's one thing for my team to just use this idle capacity, but
at Snap, we wanted to make sure even as the online need for GPUs increased, as our AI footprint
increased, we could, we should still have any team at Snap be able to leverage that capacity
for any of their needs as available.
And then we had to also acknowledge that if a user wanted to see fresh spotlight content, it
supersedes GPU need for experimentation.
Yeah.
You know, preemption had to be built in.
Yes.
Yeah.
So if, if, if, if, uh, if we had a sudden spike in traffic, we had to give up GPU capacity.
So with all of that in mind, we built out a platform ground up.
Okay.
And then, um, then we started migrating and that's, that's the, uh,
and, and we had a lot of blockers along the way and the team got really creative.
Right.
Yeah.
It was, it was a phenomenal journey.
Amazing.
Yeah.
And so you're also running an accelerated, uh, Apache Spark pipeline.
Yes.
Uh, a lot of our pipelines, uh, at a high level, our pipelines are split into, uh, daily
and hourly.
Okay.
Uh, cadence.
So hourly is mostly for guard railing.
Like I said, like, you know, we, we don't want to break user's experience no matter what.
And having that hourly feedback cycle goes a long way in doing that.
Yeah.
And then we also have daily pipelines, which serve as the statistical authority, uh, for
decision making.
So, um, our first migration to GKE plus, uh, Nvidia's sparks, uh, Spark Rapids was, uh, the
hourly pipeline, uh, because, you know, speed matted there far more, right?
So we migrated and then we migrated, uh, and operationalized it.
And during that process, we ran into a few, um, corner cases, you know, if the GPU capacity
wasn't available at like 11:00 AM when everybody was active on snap, right?
What do we do?
So we had to figure out how to gracefully fall back from, uh, GPUs to CPUs, right?
And then if the, uh, uh, shared GKE resources itself was the constraint, then we had to gracefully
fall back from CPUs to data pro clusters.
So building all of that with operational reliability in mind was also great.
Yeah.
Um, looking back on it, what learnings would you, you know, if there's a listener out there
who's embarking on a similar project or trying to figure out maybe there's a, you know, like
you said, kind of an, uh, a daily cycle of when the GPUs are in use for inference and when
they're not, they're thinking about, you know, borrowing GPUs from other parts of the company.
Learnings you would share from this whole process?
Is there a big takeaway, something that surprised you?
Right.
So, um, the direction that, uh, NVIDIA is headed in is phenomenal for these kinds of needs.
Uh, you know, NVIDIA Spark Rapids, like I said, zero code written.
Yeah.
Zero code changed.
Amazing.
So we had to figure out the image building and, uh, environment difference and whatnot.
But testing cycles, obviously any, any production workload needs to go through that rigorous rollout
process.
So everybody needs to pay attention to it.
Um, but this is a real possibility, you know, um, the NVIDIA direction.
The other thing, uh, that, uh, NVIDIA offered that really helped us a lot was NVIDIA Ether.
It's, uh, uh, it's another, uh, solution that, um, gives us, uh, spark tuning out of the box.
Okay.
Because especially when we had this fallback mechanism in place, where we had to go from
GPUs to CPUs to data proc, the environments are different, uh, the spark parameters had to be
different.
So, uh, something like NVIDIA Ether giving us a starting point and making sure the tuning
stayed consistent across all of these versions was also very helpful.
So you've mentioned obviously the, the work with NVIDIA and Google Cloud as well.
Um, kind of from, uh, taking a step back, sort of bigger picture, what are these partnerships
and working, you know, hand in hand so closely with Google Cloud, with NVIDIA, what is that
doing to the way that you and Snap see your roadmaps for both data and AI kind of growing
going forward?
Yeah.
It's, uh, I mean, huge props to the NVIDIA team and the Google Cloud team.
Honestly, it's a, it's been a phenomenal three-way partnership like I've never seen in my
career before.
Amazing.
Yeah.
It was, it was phenomenal.
And, and, and the impact speaks for itself, right?
Like we were able to cut almost, um, uh, about 76% of our job costs as a result of this migration.
76?
76.
It's, it's phenomenal.
Yeah.
Right.
So, um, that is one of the biggest headaches any, you know, uh, data pipeline at scale runs into.
So, phenomenal results.
The results speak for themselves.
Um, so without the partnership, you know, that is one of the biggest headaches any, you know,
data pipeline at scale runs into.
So phenomenal results.
The results speak for themselves.
Um, so without the partnership, you know, that is one of the biggest headaches any, you know,
uh, data pipeline at scale runs into.
So phenomenal results.
The results speak for themselves.
Um, so without the partnership, this would not have been possible in the timescale that
it was possible.
Right?
Like migrating a production, uh, pipeline with 10 plus petabytes from, uh, you know, prototyping,
exploration to full production.
results the results speak for themselves so without the partnership this would not have been possible
in the time scale that it was possible right like migrating a production pipeline with 10 plus
petabytes from you know prototyping exploration to full production in a matter of about eight to
nine months is is phenomenal right and and without the continuous uh you know back and forth and you
know knowledge sharing and partnership uh across these three companies this wouldn't have been
possible that's that's great yeah and and and in terms of the uh roadmap it it's it definitely had
an impact like i said uh my team built this bottom-up data platform to to enable any team at snap to
leverage the gpu capacity and you know uh what nvidia libraries have to offer and uh that's all we're
already seeing movement with it right even my own team started migrating other things that we haven't
even tried out so far experimenting with them you know trying out because even if we don't have ideal
capacity to fit all of our workloads all the time if we can schedule things creatively if we can move
things around we can maximize the capacity as much as we can and a lot of other teams are also picking
this out yeah it's fantastic um so you've been at snap for eight years is that right seven most two yes
state okay and snap's been around for about 15 years yes your take um working at a social media a huge
social media platform um over this span of time where social media has just you know become such a
such a core part of the fabric of so many people's lives um what's it been like to be at snap and to see
the changes both you know i i said at the beginning right i remember the spectacles that's my first
thought of snap and obviously now snapchat you know same same lineage same philosophy different product
obviously right but what's it like to just have seen the evolution of social media and then also so
so many technological changes that impact you know what you're able to do and how you do it as you were
just describing what's it been like from the inside yeah it's been uh it's been uh unbelievable of an
experience noah like um that's what gets me up in the morning every day you know like uh snap i mean if
if in in the in the in the visual communication ar ai landscape snap is has had a massive impact
on the planet yes um honestly and um having a direct role to play in it is is a great feeling right
i've seen the company grow from um you know the the camera messaging uh you know picture messaging to
what it is today uh ar uh stories which is something we invented and the whole world uh including some
newspapers so uh the stories as a format and then uh to your point about spectacles we did it before
anybody else was even thinking about it you know uh so so the company is innovative we come up with so
many new things and um running platforms inside means that i have to you know figure out a way to
enable all of this even as the company evolves and that's been having a front row seat to that evolution
and playing a big part of it has been very fulfilling fantastic um proof for the listeners viewers who
there's some out there who haven't used snapchat before for anyone who wants to get the experience
but also to learn more about about snap and maybe about some of the technical work that you're
doing are there obviously the website there's social media is there a research blog where can where can
people go absolutely so we have an engineering blog that's pretty active we share a lot of phenomenal
work that that engineers in the company are working on and um uh you know we are also um participating in
events like this and sharing our knowledge with the world so you know uh and and snapchat if you haven't
used it you should definitely give it a try it's it's uh it's different from social media i i this is
true story i got a snap from my younger son maybe 45 minutes before we sat down to do this and it made my day
so so absolutely if you haven't yeah um prove it thank you so much this has been a great conversation
and i'm sure the developers the engineers in the audience hopefully have taken a lot from it um but
thank you so much for taking the time to join us and all the best to you and everybody at snap to uh
keep changing the world for the better thank you so much thanks for having me appreciate it