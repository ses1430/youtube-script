---
title: Code with Claude 2026: Opening Keynote
uploader: Claude
channel: Claude
channel_url: https://www.youtube.com/channel/UCV03SRZXJEz-hchIAogeJOg
duration: 2849
upload_date: 20260506
webpage_url: https://www.youtube.com/watch?v=GMIWm5y90xA
id: GMIWm5y90xA
categories:
  - Science & Technology
---

# Code with Claude 2026: Opening Keynote

I'll see you next time.
Please, welcome to the stage Chief Product Officer of Anthropic, Ami Vora.
Good morning, everyone.
It is such a pleasure to see you all.
Thank you for joining us.
When I think about why I'm here today, I go back to the first time I wrote a computer program and it worked.
I didn't grow up coding.
I grew up in the foothills of the Appalachians.
I never built my own computer.
I didn't even play video games.
The first time I actually tried to build anything complicated was in my college computer science classes.
Now, this was so long ago, we had to wait in line to log directly into the servers because they were the only thing powerful enough to handle our ray tracers.
This might be familiar to some of you.
The hum of servers, the smell of old pizza, coffee, and that very specific aroma of a windowless basement computer lab.
But I can still remember that feeling of hitting compile and waiting to see if my program worked.
That feeling of joy, discovery, a little relief, and the excitement that I had made something that had never existed in the world before.
That's what hooked me.
And that's why I'm here today.
But so much has changed.
What I could only get by waiting in line at a college computer science lab, that's available to anyone, any day of the week, anywhere in the world.
No wine, no weird smells, no barriers.
Just that same feeling of excitement, joy, and relief.
And I know a lot of you feel the same way.
People say to me all the time, I feel like Claude has given me superpowers.
It's one of my favorite things to hear.
And we're seeing people use those powers.
For instance, Scott McVicker runs developer infra at Stripe.
One of his teams had 50,000 lines of Scala that needed to become Java before they could upgrade their JDK.
The initial estimate was 10 engineering weeks.
They used Claude and finished in four days.
And sometimes speed isn't just about efficiency.
It's about what's waiting on the other side.
Felicia Kukuru is the co-founder and CEO of Binti.
Her software runs the systems that caseworkers use to place kids in foster care.
The paperwork, the home visits, the licensing process.
This year, her team used the Claude API to give caseworkers back hours they used to spend on paperwork.
That took 20 days off the process of licensing a foster family.
20 days.
It's not just an efficiency metric.
That's a kid connecting with a family.
And that excitement, joy, relief, that feeling of discovery, it's something I hear from everyone.
I'm going to guess, though, that everyone here experiences it differently.
Some of you are living on the frontier every day.
Some of you are bringing along the people around you.
And some of you came here because, like me, you can feel the ground shifting under us,
and you want a view of what's to come.
Trust me, I feel all of those things often in the same morning.
I come to work with a plan, and then I have to tear it up by lunchtime because something new has happened.
Does that sound familiar?
And, you know, that makes sense when we step back and look at how fast these models are getting better.
At Anthropik, we talk a lot about the exponential, and I think that's what we're all feeling right now.
Remember, a couple years ago, the frontier of model development was something so good
it could write a decent email, and we were pretty happy about that.
A year ago, we were standing on this stage.
Opus 4 was the headline, and the idea that an agent could run for an hour without a human checking in
felt like a stretch goal.
But then six months ago, agents were running end-to-end overnight, so we'd wake up to finished work.
And then last month, Mythos read the entire OpenBSD source tree and found a 27-year-old vulnerability
that had survived every human reviewer, every fuzzer, every static analyzer thrown at it for almost three decades.
The jumps keep getting bigger, and the intervals keep getting shorter.
But even though model capabilities are improving on an exponential, most organizations are still adopting AI on a linear path.
And that means there's a gap between what AI can do and what it's actually doing for people.
Closing that gap, translating model capability into something real people use to solve their problems.
That's what developers do.
That's what you all are doing.
And we're seeing it happen.
Year-over-year, API volume is up nearly 17x on the Cloud platform.
And on Cloud Code, the average developer is now spending 20 hours per week running Cloud.
Now, like you, we've been shipping a lot lately, and we want you to walk away from today with a clear picture of where we're headed.
So you can plan for it and ride the exponential with us.
Let me say up front, we don't have a new model to unveil.
Today is about how we're making our products work better for you.
So you can close the gap for the rest of the world.
And this morning, we'll show you what that looks like.
First, Diane will talk about our foundation, the model layer.
She'll share more about our frontier models and what's coming.
On the Cloud platform, we're shipping updates to Cloud managed agents, outcomes, dreaming, multi-agent orchestration.
And Angela and Caitlin will walk you through how the platform handles the infrastructure so you don't have to.
And on Cloud Code, Kat and Boris will walk you through how you can use new primitives, like routines, to let Cloud Code prompt itself, even when you're away from your computer.
But all of this comes back to you and what you're going to build.
Because most people will never call the Cloud API.
They'll never open a terminal and type Cloud.
They'll experience AI through something one of you built on the Cloud platform.
Whether that's a designer exploring new directions with Canva.
Or a lawyer getting a brief out the door faster with Legora.
Or a developer using any one of the world's best coding agents.
So thank you.
You all shape what AI feels like for everyone else.
We'd never be able to build everything that people need to solve their problems.
That's something only you all can do.
And one way we want to show our gratitude is by sharing a little exciting news.
As of today, we are increasing rate limits for developers on Cloud Code and the Cloud platform to help you keep building and closing that gap for the world.
More specifically, we are doubling Cloud Code's five-hour rate limits for pro, max, team, and seat-based enterprise plans.
And we're raising our API limits considerably for Cloud Opus.
We're making this possible by expanding our compute partnerships.
We're partnering with SpaceX to use all the capacity of their Colossus One data center.
And we're investing this directly into individual developers and small teams.
Over time, we'll continue to explore every way to help you get the best out of Cloud.
So, including our existing compute efforts and even bolder bets.
So, thanks for being here today.
Thanks for partnering with us to shape what AI looks like for the world.
Thanks for giving people superpowers.
Up next, Diane, who leads our research PM team.
Thank you.
Thanks, Sonmi.
I'm Diane, and I joined Anthropic back in 2023.
And I've been a part of every model since Cloud 2.
For those of you who are counting, that's bringing 18 versions of Cloud across Haiku, Sonnet, Opus,
and now Mythos to users and developers like you.
We wrestled with making Opus 3 great at adhering to JSON and also making it the best at writing long-form code.
With Sonnet 3.5 Mu, or as we all now finally know it, Sonnet 3.6, we taught Cloud to make and use a computer safely.
And with Sonnet 3.7, which had a tendency to be slightly too over-eager, we figured out the right way to expose that to users and developers so you could get the most out of Cloud.
This time last year, we used Cloud 4 to be able to use thinking dials in a way that worked well and to adjust test time compute.
And we haven't slowed down.
In the last 12 months, we shipped eight frontier models to developers and users, each one building upon the last, allowing you to write better code and the products you build go further than previously before.
The model layer underpins everything else you'll hear about today.
And that's the bottom line.
As model intelligence increases, your starting line moves forward, and you could do more than ever before.
We talk about the exponential a lot at Anthropic.
You heard it a little bit from Ami as well.
For me, the exponential means that as model intelligence increase, the use cases you can build and deliver to your users increases exponentially.
For example, agentic coding is far more impactful than code autocomplete.
And in this way, new products and new experiences create new markets and grow the pie for everyone.
In research, we don't think about the exponential as sweep-inch numbers going up.
It's also about creating and tracking capabilities that previously didn't exist until we designed and created them.
Tool use, computer use, thinking that adapts to the problem.
Agentic loops that hold a plan over hundreds or thousands of steps.
And long context windows that teach Cloud knowledge that it previously didn't have.
And these capabilities don't just stop at code.
Today, Cloud can generate and iterate on visual designs, analyze and create complex work deliverables, and also navigate business domains you might be a part of in open-ended, ambiguous fashions.
That's because the model intelligence, the core foundation, has gotten smart enough and strong enough to support all of this.
When you're building on Cloud, you're building on the model family that created these capabilities first and has had the most time to make them reliable.
Let me make that concrete with our latest model, Opus 4.7.
Amp, the coding agent, moved their entire smart mode onto Opus 4.7 because they scored the highest on their benchmarks, and they were able to simplify their tooling, change their scaffold, because the model no longer needed the help.
Rakuten ran it on their benchmarks and resolved three times the number of production engineering tasks than they previously could.
And finally, Intuit saw Opus 4.7 identify its own logical faults during the planning stage, figure out what was wrong, backtrack, resolve it, ultimately leading to faster and cleaner execution.
The day after we launched Opus 4.7, we launched Cloud Design by Anthropic Labs, one of my favorite launches this year.
Already, people are building production interfaces with a combination of Cloud Design and Cloud Code.
This is because Opus 4.7 has a real taste for visual design, the right nuances to show while adhering to your design principles.
We also hear from everyday users that people like to use Cloud because it understands the full assignment and can figure out when to push back and question assumptions.
At the same time, as each of us already know, having built on these systems, the models are unfinished and their work's in progress.
They can still be stomped sometimes by very basic things and also lose the thread when you introduce a lot of context.
That's what makes this exciting and thanks for being on this journey with us.
Here's a little about what we're working on and what's ahead.
First, higher judgment and better quality code taste.
This means versions of Cloud that you can trust with complex, autonomous engineering work.
Second, context windows that feel infinite when combined with high-quality memory.
So it feels like you could do a long-running task while getting better results.
And finally, multi-agents coordination, powering teams of agents and instances of Cloud that collaborate on big goals that are far too big for any single instance ever could.
The way I think about progress and model intelligence is task horizon, which is a measure of how long a version of Cloud or a model can work autonomously
while improving on its deliverables and the quality of its work.
Last time this year, models can work for minutes.
Now, most of you and I probably have agents that are running for hours on end.
And tomorrow, we'll have agents that are proactive, always on, and know what to work on without losing the thread.
So what do we as developers make of all of this?
The exponential will keep improving.
And you need to build for emerging capabilities and not just for today's versions of Cloud.
This is because new models would be far more capable than the ones we have access to today.
It used to be that we have to build scaffolding to keep every version of Cloud up.
And now scaffolding is there to actually amplify model intelligence.
You used to have to design complex iterative loops, give it the right tools, figure out how to do retries.
And now all of that can be folded into the right thinking and the right execution right within the model.
You are already seeing where this can go.
Opus preview, missus, is the next point of that exponential.
And it's not a small step.
Therefore, how we all work with Model and Cloud need to change.
Here are some things we think about at Anthropic.
First, you need to design for the next version of Cloud, not just the current one.
We've seen countless times that the developers who win are the ones who optimize their architectures to absorb the next intelligence jump, not just today's incremental accuracy.
This means maintaining and creating harder evals, building ambitious prototypes that you don't think might work today.
Because that's how you'll notice when the exponential is improving and moving under you.
That something that previously didn't work all of a sudden starts passing.
That's a sign that you probably have something magical to give to your users that didn't work before.
And here's what the teams that are getting the most out of Cloud have figured out.
Model upgrades are a business opportunity.
The teams that are getting the most out of Cloud models are the ones who make upgrade cheap.
This is automated evals, simple scaffolding, and ambitious prototypes and uses of capabilities that others haven't imagined yet.
We believe that the exponential on slide one is going to keep looking like that.
As model intelligence increase, you as developers have the chance to make a head start to experiment with building new use cases, creating exciting new products for your users, and ultimately creating new designs and markets and ultimately growing the pie.
Everything Katelyn and Angela are about to show you will give you the tooling to make all of this possible and come to life.
Thank you so much for being here.
Thank you so much.
So model capabilities are on the exponential.
But businesses are still operating on the linear.
And so as a business, it's never been more important to make sure that you're really able to harness the power of that exponential.
But what's stopping businesses from really snapping to that?
Well, it really boils down to two key problems.
The first one is getting the right outcomes.
Getting the right outcomes is still too difficult, right?
You have to be able to do a lot of prompt optimization, tool construction, harness engineering.
There's still a lot of work that it takes to really steer the model to exactly where you need it to go.
That's right.
And the second problem is you want to ship fast, but you want to ship scalably at the same time.
Everyone in the tech space is moving insanely fast right now, and you've got to keep up.
But to win, you need quality, too.
It's really easy to ship prototypes, but it's really hard to scale in production.
So we've built the Claude platform to give you everything you need to get those great outcomes and to ship with speed and scale at the same time.
The platform comes with API primitives that are tuned to Claude models.
It gives you infrastructure to build and scale agentic systems.
And it gives you controls to operate those systems.
That's right.
So if we go back to our problem statements that we hear all the time from different businesses,
one of the most common ones that Kayla and I hear all the time is how much businesses need high intelligence but, of course, at lower cost.
So one of the ways that we're solving this is with the advisor strategy.
This is really easy to implement.
All you have to do is update your tools array inside the messages API.
And what we're doing is that we're actually providing an agent architecture that splits execution from advising.
So in execution, you can pick a smaller model.
It will be a little cheaper.
But then that small model, when it needs advice on what to do next, it can actually reach out to a bigger model for help.
Yeah, so in practice, this means you could use a haiku or a sonnet class model to execute an Opus as an advisor.
And when we ran this with Sonnet executing and Opus advising, we saw Sonnet perform way better than Sonnet alone.
But more importantly, we saw Sonnet perform even more cheaply than Sonnet alone because Opus was able to advise it to get its work done better.
A great example of this was EveLegal.
EveLegal used the advisor strategy, and they told us they got frontier model quality at five times lower cost.
Yeah, and that's awesome.
We love stuff like this because you can go ahead and use this in things like a freemium model, right?
When you offer these kinds of freemium experience to your users, you have to be conscious of the costs that you're going to incur.
But obviously, you want to make sure you have a good experience for them.
It's also really great for areas where you have an extremely high volume amount of workloads.
And of course, you'd have to be a little conscious of your ROI.
So that's all great, but what about speed and scale, those two things that Caitlin mentioned to be really difficult to achieve at the same time?
Well, most recently, we introduced Claude Managed Agents.
Claude Managed Agents is an agentic harness that's paired with production-grade infrastructure.
Teams are able to go from prototype to production in literally a matter of days.
The teams that we've worked with have been able to ship literally ten times faster with Managed Agents.
Another great thing with Managed Agents that we love is that it bundles in a lot of the best practices out of the box.
So for example, when you build an agent, one of the best practices that you want to make sure that you do is, of course, give it memory.
That way, the agent persists user preferences, and it kind of remembers closer to what you want it to do every single session.
It's a little difficult to build memory, and so this is an example of a best practice that we actually just bundle in out of the box, and it's automatically tuned for Claude.
And we want to make sure that everyone hears this.
When we do give you memory, that memory is ultimately yours, so you can take that and import it wherever you'd like.
So one of our favorite examples of someone who built on Managed Agents was Notion.
Notion wanted to build for speed and scale at the same time, so they chose to build on Managed Agents.
And they built the ability for you to fire off Claude Agents directly within their product experience for long-running, complex, autonomous tasks.
Yeah, and we love that feature, that's super cool.
All right, well, today, we are upgrading Claude Managed Agents with three really powerful features.
We're going to be introducing multi-agent orchestration so that you can actually create fleets of agents to solve really complex tasks.
We're going to introduce outcomes, which allow you to specify exactly what success looks like.
And then Claude will literally just iterate until it gets it done.
And we're going to introduce dreaming.
And this one we're really excited about.
But with dreaming, Claude is actually able to self-learn.
It's able to actually inspect over its previous sessions, figure out skills that it missed, lessons it should have learned,
and actually apply those directly to memory on its own.
But instead of just talking about all of these with you, we're actually going to show you live what these things look like.
So Caitlin, let's do that.
Let's do it.
So Caitlin and I have been inspired by some of the announcements earlier today.
There's greater API rate limits for Opus.
And we've been hanging out with a particular space company for most recently.
So we've been inspired to create a little startup of our own, a fictional one to be clear, called Lumara.
And with Lumara, we decided, you know, why don't we build agentic software that helps us autonomously land drones on the moon.
And we really care about speed and scale at the same time.
So obviously, we are going to build this on Claude Managed Agents.
Exactly.
So let's say we lined up our first customer.
And this hypothetical first customer wants to land drones on the moon to mine for hypothetical materials.
And this is a big, ambitious job.
And despite all of our dreams, we're not actually aerospace engineers.
So we're going to need really awesome agents to get this work done for us.
So we're going to integrate all three of the new features that we just talked about.
And we did this for our first customer.
And I'm going to show you, using the Claude API CLI, how we actually set that up.
So first things first, this is a big job.
So we actually want multiple agents to help us get this work done.
So I'm going to go ahead and show you the agents that we actually have set up for our customer.
The first one, we have a commander agent.
And really, the commander agent's job is to make sure that this whole mission goes well.
Then we've got a detector agent.
And the detector agent's job is to make sure that we're actually finding the sites to land on
that will have high-quality mining materials.
And then we've got our navigator agent.
And the navigator is making sure that we're landing our drones safely and flying them to their destination.
So I'm going to go ahead and actually set up our commander to be a coordinator of the other two agents.
And when this is running, what's actually happening is that the commander is spinning up a session.
And then for each of these sub-agents, they have their own independent threads so that they have independent context windows.
This is a very intentional design.
And we found that by doing this all together and then merging in all the results, we get better performance.
Exactly.
So that's multi-agent.
Let's go ahead and integrate outcomes.
So the way outcomes work is we want to make sure that our customer, which has very specific criteria for what they want to accomplish,
can define that criteria.
And then we can provision a greater agent that actually makes sure we get that outcome.
And so outcomes actually just start with a pretty simple markdown file.
So here you can see a markdown file, again, really, really simple.
And it just kind of outlines this criteria that shows us whether a run is successful.
So we want our drones to touch down softly.
We want them to land on clear ground.
And pretty importantly, actually, we want to have enough fuel in reserve so that we can get our drones safely back to Earth.
So in order to actually set out this rubric for our outcomes, I'm going to go ahead and send an event to our session that defines this rubric as our outcomes.
Yeah, and when this is running, like Caitlin mentioned, we actually create a separate grader.
And this grader agent is evaluating across the session as to whether or not, in each run, we've actually met the rubric that was specified.
Now, of course, it could one-shot this, but more likely than not, it probably needs to iterate over a couple sessions to do it.
And as Caitlin has highlighted here, you can specify the max number of iterations you want to allow this to do.
So we've integrated multi-agent, we've integrated outcomes, and it's time to test.
Our customer gave us some data on six hypothetical sites that they might want to land their drones on, for us to run some simulation sessions and see what happens.
So I'm going to pop over into Lumara's dashboard, where you can see that I ran a simulation against these six sites.
Yeah, and this is pretty good.
I mean, this is a one-shot with our whole system.
It's got the multi-agent architecture.
It's got the outcomes feature integrated.
And you can see here that it solved four out of six sites correctly.
But clearly, it could have done a bit better on sites three and sites four.
And like any two good founders, we obviously want to hill climb on this system.
And normally, hill climbing is a pretty difficult process.
You have to put a lot of work together to go and do that.
But we're going to show you how we are hill climbing on this with just dreaming.
Yeah, so we ran the simulation yesterday.
We weren't quite happy with our results.
And we came into the Claw developer console into our dreaming interface.
And you can see I can actually just hit this little button that says dream, choose a memory store where a dreaming agent can go and look over all of those past simulation sessions and write its learning to memory so that all of our new sessions can actually reference those learnings in memory to do a better job.
So I did this last night.
And this is our dream that ran.
And you can see that we wrote a bunch of stuff to memory, which is awesome.
And crucially and most importantly, the agent actually chose to write a descent playbook so that all of our additional sessions that we run going forward will have this playbook to reference, including all sorts of heuristics from the different missions that it ran previously.
This is like a really robust playbook of a whole bunch of information that we can go ahead and use.
So this ran overnight last night.
And I popped back into Lumar's dashboard this morning and ran a new simulation now that our system is upgraded.
And that's awesome.
We were able to actually hill climb and not regress on any of the ones that we cared about.
But the two sites that could have improved have actually improved.
And to hill climb on this, all we had to do was just have Caitlyn press a button in the console called dream.
All right, awesome.
Let's wrap up.
And everything that we showed you here today live in this demo is available on the Cloud platform for you all to build on.
Multi-agent orchestration, outcomes, and dreaming are now making the Cloud Managed Agents primitive so much more powerful so that you can use it to construct deep, powerful agentic systems that scale.
Whether you're landing drones, autonomously on the moon, or you're building the next big business, Cloud Managed Agents is here to help you construct the agentic systems that you need
and solve the problems that you're looking to solve.
Now, we'd love to pass it on to Kat and Boris, who will show you how Cloud Code is making it even more fun to build as a developer.
Thank you.
Angela and Caitlin just showed you how the Cloud platform closes the gap between what models can do and what agents' businesses ship.
We have a related challenge on Cloud Code.
We also want to close the gap between model capabilities and what every developer can actually do with them.
First, I just want to thank all the developers here in this room.
Thank you for trusting Cloud Code on your production databases back when Sonnet 3.7 was our frontier model.
And when our product was a bit rough around the edges.
Your support is what makes our team so excited to come in every day and make Cloud Code even better.
Let's start with why Cloud Code exists.
Software development is being reinvented in real time.
The mission of Cloud Code is to close this gap between the great ideas that you all have and shipping a product to the market.
And the way that we enable this is we build tools that elicit the frontier intelligence from our models and we make them accessible to every builder.
And we don't think of ourselves as having a finished roadmap for you.
We think of ourselves more like mountaineers, kind of like climbing alongside you into terrain that none of us have explored before.
Learning together what works as we go.
And we're growing with you.
We're growing with increasing AI capabilities.
And we're navigating these new challenges together.
I still remember a year ago when I would give Cloud Code a task.
And I would carefully review every single edit it was trying to make.
Every single permission prompt giving it really detailed feedback on what I liked and what I didn't.
And just like holding its hand every step of the way until the result was good.
I remember some of these tasks would be like 100, 200 permission prompts until I got the final result.
And now most of you are running in auto mode.
You're delegating permissions to Cloud.
And you're checking in after Cloud has done a lot of its work and has a PR for you to review.
Over the last year, we've expanded the number of ways that you can use Cloud.
We started with the terminal.
Then we launched the IDE.
And now we have desktop.
We started with the CLI.
This is still the interface for power users who want a minimal text interface, who want all the latest customizations and the most control.
Then we added IDE because a lot of you actually want the same powerful agents, but you want to follow along with all the code changes that it's making.
And then based on all of your feedback or something a bit more visual, we knew where we had to go next.
We launched our newest surface, Cloud Code on Desktop.
It's a surface that's designed for people who want this, like, full screen graphical interface, built-in previews so you can watch as Cloud develops your app,
a sidebar control plane for all your agents, and the ability to render images and rich outputs.
We've built desktop to be a control plane not only for your local sessions, but for your remote ones as well, with visual indicators for which agents are stuck and which ones are ready to go.
The IDE and the desktop app are built on the Cloud Agent SDK, the same SDK that many of you are already building on.
Many enterprises have adopted Quad Code tools wall-to-wall.
At Anthropic, this has driven a 200% increase in the number of PRs per engineer, while keeping the same code quality bar,
even as our engineering team has scaled substantially.
Together with you all, we're discovering and redefining the future of what software engineering looks like.
We're embracing these new challenges by embracing automations powered by Quad to overcome each.
I'm going to walk through a few of them right now.
Here's the feedback that we heard from our users and what we've built with the help of this community.
We heard from you that you want to spend less time on code review.
So we shift code review that deploys a team of agents to catch critical bugs on your behalf.
Thousands of companies use this every day, including all internal Anthropic teams.
We heard from you that you really want to code on the go.
So we launched remote control and we added Quad code to the iOS and Android Quad apps so that you can fire off a task from anywhere.
You're no longer walking around with an open laptop,
balancing it, trying not to fall.
And you're no longer stuck at your desk.
You can now go to a park, touch grass, and still code.
We heard that you're spending a lot of time babysitting PRs,
fixing flaky CI tests, addressing code review comments, resolving all the merge conflicts.
So we added autofix.
It just listens to all these events, proactively puts up fixes so that your PRs are always green.
We heard from you that you're kicking off Quad code tasks on new tickets and new customer bug reports.
So we thought we should build routines.
So we have routines and it lets you configure once, listen for web hooks, API events,
or run on the schedule, and it will just kick off Quad code automatically for you.
So instead of you having to manually kick things off, Quad will handle it.
And last, we heard from you that you're launching so many features that your security teams are having a hard time keeping up.
So we built Quad security.
It scans your whole code base overnight and it can kick off Quad code to address the vulnerabilities that it finds.
All of these primitives compose together and this helps all of us together adapt to the future of what engineering looks like.
Everything I've covered is something that you can pick up today.
It's been especially exciting to see how a range of companies have taken these tools and adopted them at the scale of entire orgs.
First, I wanted to share about Shopify.
They power e-commerce for millions of merchants worldwide and they've imbued AI across the entire engineering org and changed their culture.
They're using Quad code across the company, both on engineering teams, but also non-engineering, so design, product, data science.
They're building it directly into their platform and standing up tools at scale.
Andrew McNamara is the director of Applied AI at Shopify and in his words, the speed is just crazy.
Quad code has completely transformed how they build internal tools.
Another example is MercadoLibre.
They're Latin America's most popular e-commerce platform.
They serve over a hundred million buyers.
Their org is 23,000 engineers and everyone runs on Quad code.
When that happens across an org, the work itself changes shape.
Engineers are pointing agents at tech debt that people haven't touched in a long time and people don't have time for.
It's reviewed more than 500,000 PRs with human oversight and modernized more than 9,000 apps.
Oscar Mullen, who leads technology, is aiming for 90% autonomous coding and fully agent-driven PR loop by Q3 of this year.
And we hear this from many others across the industry.
The detail I love the most here actually isn't this number.
It's that a lot of the managers and VPs we talk to are like getting their hands dirty in the code base again.
Quad code is putting coding back in the hands of people who've spent the last decades on roadmaps and reviews.
And now they're back building.
We see this across the industry.
Millions of developers are getting more products shipped at a higher quality than before.
Now, let's see what this actually looks like in practice.
To take you through it, please welcome the head of quad code, Boris Cherney.
Thanks, Kat.
Can we do a quick selfie?
Phew.
All right.
Before I jump into this demo, I just want to mention something.
Everything that we're showing today still feels magical to me, and I work on quad code every day.
Even in Anthropic, we share screenshots back and forth of the cool things that people are building with quad and things that people are doing in the wild.
And honestly, I just feel excited to be on this journey together, discovering all of this.
So today, I'm excited to share a few more examples of what this looks like.
Unfortunately, we can't all work in the lunar drone business.
So for this demo, let's imagine that we're an engineer at Acme Pay, and it's a payments infrastructure company.
We're going to start the quad desktop app, and we're going to start by working on a single task.
In this session, Claude's working on adding refunds to Acme's merchant dashboard.
It's building a full implementation, item potency, so a duplicate webhook doesn't double refund a merchant.
There is multi-currency handling across all the regions Acme serves, and audit logging for the compliance team.
It writes the implementation, and it's going to verify its own work.
Claude pulls up the merchant dashboard, it triggers a refund, and there's no success toast.
Claude sees the failure, it traces it back to a race condition in the optimistic update.
It fixes it, and it's going to verify that it actually works in a browser before it calls the task done.
Now, let's zoom out.
This session wasn't running alone.
It's actually one of many sessions that were all running in parallel and being managed in parallel.
In the Claude desktop app, you can now see all your Claude code sessions.
Which ones are running, which ones need your input, and which have PRs that have already been merged and closed.
Synchronous coding is now just a slice of what's happening at any given moment, and we think that going forward a lot more code is going to start to be written in an async way.
And this is why we keep talking about verification.
If Claude can check its work, you can just let it run while you work on something else, and you come back to a fully working result.
And for me personally, a lot of my code nowadays is written by routines.
I'm not the one doing the prompting.
I'm the one creating a routine that does the prompting.
For engineers in the room, think of it like a higher order function.
Routines are a higher order prompt.
For example, the refund session that we just looked at.
A teammate filed a GitHub issue overnight.
A routine watching the repo picked it up async and then kicked off the work in Claude.
With routines, developers can set up async automations and wake up to PRs that are ready to merge.
Here's our routines view.
Routines can be run on a schedule.
They can be kicked off by web hooks.
Or they can even be kicked off by arbitrary API calls.
You can run them locally on your machine or on remote cloud compute.
Let's look at one more feature.
This is CI autofix that Kat talked about earlier.
And what it's doing is it's watching the PR the prior session just opened.
Its job is to babysit the PR to get it all the way to production.
It's going to autofix any comments from code review and security review.
It's going to autofix CI.
And it's going to autorebase if there's merge conflicts.
And look at what just happened.
CI flaked on a network timeout.
The routine woke up.
It diagnosed it as a known infra issue.
It retried the job and now it's green.
And actually in the cloud code code base, we have it not just retry.
We have it fix the root cause every time.
The engineer who owns the PR is never going to see a red X.
And that work is off their plate.
And that's the shift.
The default isn't I'm going to prompt cloud code.
The default is now I will have cloud prompt cloud code.
Everything you just saw is available today, including routines, the latest updates to the cloud desktop app.
We're excited for you to try it out and let us know what you think.
We hope these features continue to close the gap between your ideas and shipping products.
And that's really what every talk today was pointing out.
Diane's capability curve.
Angela and Caitlin's agents that grade and improve themselves.
What Kat and I just showed you.
These are three layers of one story.
The capability is already here.
The gap left is how fast we put it to work.
I encourage you to spend the rest of today exploring these layers.
Research talks if you're evaluating the models.
Cloud platform sessions if you're building for your users.
Or cloud code workshops if you want to learn more ways to bring cloud into your day-to-day development work.
Dive in.
Go deep.
And start building with us.
Thank you.
And start building with us.
Thank you.
you