---
title: ElevenLabs' Mati Staniszewski: How Voice Becomes the Interface for AI
uploader: Sequoia Capital
channel: Sequoia Capital
channel_url: https://www.youtube.com/channel/UCWrF0oN6unbXrWsTN7RctTw
duration: 1609
upload_date: 20260506
webpage_url: https://www.youtube.com/watch?v=ZNzYN2jyVTU
id: ZNzYN2jyVTU
categories:
  - Science & Technology
---

# ElevenLabs' Mati Staniszewski: How Voice Becomes the Interface for AI

So I love line charts and bar graphs as much as the next guy, probably more.
The story of Eleven Labs is also interesting from a human perspective,
which is you started a company with a childhood friend.
So maybe take us back to 2022 or earlier,
and just tell the human side of the Eleven Labs story to start.
I have the most luck in the story of Eleven Labs
because, well, it started in 2022, it feels like it started 17 years ago
when I met my co-founder, Piotr.
All the names in Polish are complicated, luckily, for us.
But we met in high school, became best friends,
took all the same classes together,
and then through the years did everything together.
So we traveled together, studied together, worked together,
and time is on our side. We are still best friends.
It's working out.
And part of what started Eleven Labs is inspiration from where we are both from.
We are both from Poland, suburbs of Warsaw.
And there's a very peculiar thing in Poland.
If you watch any foreign movie in Polish language,
all the voices, whether that's a male voice or a female voice,
get narrated with one single character.
So as you can imagine, pretty terrible experience.
You have literally one voice narrating everything.
It usually also, on purpose, is kept in monotone,
so you are meant to interpret your own emotions for that content.
And while we grew up with this, this is still happening today
for majority of content.
And that kind of opened our eyes into one of the clear things
across the domain, across audio domain, across the future,
will be this ability for everybody to speak any language
with the same emotion, the same intonation.
And we started diving deeper into that problem
and realized the problem of audio exists in so many other domains too,
whether that's narrating the content around us,
whether that's the books not being available in audio form,
whether that's the news articles that we could read,
whether that's that language barrier,
or in the future, as we heard in the previous conversations,
the future where human noise, the robots are around us,
the voice will be the primary interface to a lot of that technology
and something we would love to fix and solve.
Excellent.
And Eleven Labs builds frontier models for audio.
I think there's a paradigm now where to build a frontier model
you have to start with hundreds of billions or billions of dollars
and then figure out the rest later.
Eleven Labs did not take that path.
May I talk a little bit about your approach towards building this company,
why this hasn't been replicated,
is that even possible in 2026, et cetera.
Yeah, that goes, I think that continues that great lack in timing
because we started in 2022.
For those of you working in the domain at the time,
that was a year of crypto and metaverse.
Nobody was still working on the AI side.
Even further, people were starting to work, of course,
on the text models, on the visual models,
but audio as a domain was still considered a big need.
There's so few researchers in the space working on that work.
So for us, that was a good part of picking that domain
where A, we were excited about where that future is called.
We felt that people around just didn't realize the value of that domain.
But three, the requirements of what you needed to solve
were very different.
The audio models were smaller.
So you don't need as much compute as you need
for some of the other sister domains.
The data needs are big, but while there's a lot of audio data,
we knew that the thing to actually get that audio working,
you will need to figure out how to transcribe a lot of that data
and annotate a lot of that data, which we knew we can do.
And then ultimately, it all boiled down to architectural side
of can we solve that part in a good way.
And here, my co-founder is one of the smartest people I know
and a great researcher and has been able to assemble
some of the best people in audio to help us.
And we took a slightly untraditional approach at the time.
We started in London.
We had a lot of people between London and Warsaw
and started a company in a completely remote way.
So we wanted to hire the best researchers wherever they were.
We were going through the classic GitHub scraping
and trying to reach people based on their work
instead of based on their presence.
And based on that work, we would reach out to those people.
We would always share our samples and try to get them
to join the team.
And that's how we assembled the first set of people
who we think are some of the best researchers in that audio domain.
And through the years, they still help us
crank a lot of those models into production.
Then we launched the product.
I think the slightly different approach we took
was monetizing very quickly.
So trying to get some of the revenue stream back
so we can find a lot of the work in the models.
We tried to stay healthy on the margins
so we can continue investing with the assumption
that it's better for us to figure out that stream
and be able to be independent in the development.
But then as the ambitions grew, we knew that we needed to train models.
So we, of course, brought a lot of money externally as well.
And I think like projecting to today, one thing that's clear for us
is there's still so many of those niches that people don't tackle
that you can start with and then step by step start opening them up.
I think a lot of customers see 11 Labs through their narrow needs, right?
Maybe take a zoomed out view.
Like what is the suite of models that 11 Labs works on?
How do you prioritize them?
How do you organize R&D, et cetera?
So we started with the first text-to-speech model.
So the model that could finally understand the context
of what's being written.
And based on that context understanding,
get the right emotion, the right intonation from text.
So it was a happy sentence, you get that happiness out.
If it's a dialogue, it can pronounce the dialogue out.
And then continuously started adding that.
So it started with the problem of breaking down language barriers.
The things you need to solve dubbing is transcription,
so understanding, then the translation,
and then text-to-speech.
So you first solve text-to-speech.
Then we knew we needed to add the other component,
which is speech-to-text and being able to transcribe content
in a great way.
Then how we combine those models together.
So that kind of was the first three models
in the first couple of years.
And then, of course, the other thing
started happening across the space,
which is that a lot of the reasoning models
started becoming quick enough and smart enough
at the same time where you could imagine
those interactive experiences being possible.
And that's where we started launching
more of the real-time streaming models across audio,
and then combining those into conversational experiences.
So we added effectively all the stack,
all the turn-taking and orchestration
to create a voice engine for a voice agent.
And then on the other side,
as we realized that the emotionality
is something we can solve,
we added some of the hardest modality in audio,
which is music and being able to produce music.
So today we span the entirety of the research of audio,
whether it's text-to-speech, speech-to-text,
combining those models together in both localization with dubbing,
with orchestration with voice engine,
and then being able to do that across music as well.
And what's the-- all those things
and all that interesting development work,
was there any "oh wow" moment in terms
of what these products are capable of that you can remember?
You know, there's so many,
and it's kind of the bar changes for all of us.
The first moment for us was--
well, the first moment for us,
they always used my voice as a testing voice
because it has this weird accent.
And the first time was like,
when we could replicate my voice based on a good sample,
that was like a first "wow" moment to myself.
And you always go through this moment like,
this is not how my voice sounds like.
And then you listen to yourself side by side,
and it's like definitely how it sounds like.
Unfortunately.
Then the second moment was where we first got it to laugh.
And people were like, okay, this is actually the thing
that makes the whole experience more human.
The laughter, the pauses, the umms, the umms,
the imperfections.
So we started getting those out,
and that was the moment for us
because we made it to the top of Hacker News
with the first AI that can laugh model,
which was a very proud moment for us.
And then, of course, through the years,
kind of that extended where--
you might remember in 2023, 2024,
there was a Javier Mele speech that went viral
where you could speak other languages
that was translated into English.
And it was the first time
where we could still hear his voice out there.
So that's the kind of continuous "wow" moment
that was something that was completely impossible.
And then we saw that happen time and time again
with Narendra Modi, with President Zelensky,
all the way to recently,
one of the, I feel like, pinnacles of the voice performance,
Matthew McConaughey giving his newsletter
and his iconic lines in Spanish and Portuguese,
where for the first time his family who speaks that language
could hear him speak those languages too.
But for most recent pieces,
the two ones that we are excited about bringing to production,
I think the first one is finally figuring out
the emotional intelligence in that interactive experience.
So in the voice agent experience,
where it doesn't only get the right intonation and emotion,
but can understand the other side.
So if somebody is stressed,
it gets and delivers that soothing, reassuring emotion.
If someone is excited, maybe it matches that.
If someone speaks slowly, it makes sure to slow down.
And that emotional intelligence is something
that we are finally seeing internally a path to solving,
which will be just a continuous step change to what's possible.
And then the second one, which will apply there,
but also apply into general audio spaces,
is audio general intelligence,
where you can combine audio models together in one stream.
So you could theoretically have a model that narrates,
then pauses, and let's say starts singing
with that same continuous voice.
And that's something that's extremely hard to combine today
and something that would be possible, I think, very, very soon.
And voice, you mentioned voice agents.
And it seems like everybody is, at least on the customer side,
everyone's buying a voice agent.
And I think intuitively you think customer support,
you know, the old phone tree replacement.
What's actually going on in the world of voice agents?
And what do you think are the most interesting,
overlooked opportunities,
spots where startup founders should focus?
Yeah, of course, the customer support is probably the one
that everybody heard and knows about very well.
I think the second thing and the second thread we are seeing
is increasing shift to revenue-generating opportunities
where voice agents can act in sales,
whether it's inbound or outbound of sales.
It doesn't replace the entire experience,
but takes and amplifies part of that experience.
Maybe a good example is Deliveroo,
where Deliveroo will have voice agents
that contact the restaurants to capture their opening times.
And based on their opening times,
they can update the riders and drivers,
and, of course, the people ordering
on when to get to that work,
all the way through to the inbound sales
where increasingly people,
that's a good example of Deutsche Telekom,
will be contacting to inquire about the service,
inquire to buy a product.
And instead of going through the dropdown,
instead of going through the form,
you can speak with the voice agent to leave that information.
We do it ourselves, too.
So we have a good metrics of an understanding
of what's happening there.
One, of course, so much simpler and quicker
to go through instead of going through that form.
But the second thing that started happening
in that inbound sales flow is we had a lot more information
that people started leaving because they would speak
about the use case they're coming with,
but then where it's not working, where it's working,
some of the other use cases that they are evaluating,
which we can combine and then just deliver
such a much better experience afterwards.
On the overlooked side, I think my favorite example
is the citizen support education and healthcare
will completely change.
On the citizen support, like all of us
would benefit from just generally better government access,
whether that's understanding how to fill in the taxes
that I think many of you went through earlier this month,
all the way through to just learning what is the policy for travel abroad
and how that might affect the space.
We've recently seen that work deployed in the government of Ukraine,
who we think is one of the most advanced governments on that front.
We traveled to Ukraine working with their team,
and what they are trying to solve is they have a government app
which every citizen can access and get information about what's happening,
but given the war, given the frontline and lack of that access,
they wanted to figure out a new channel for people to be able to call in
and get that information.
So they created a voice agent effectively where you can call in
and get the information about what's happening on the frontline.
You can get education help and some of the lectures delivered to your kids
all the way through to proactive engagement about staying safe and staying out there.
And maybe last example on education front, and that's probably my favorite one
as I think about that changing.
It's just how incredible would it be to have someone that is an incredible teacher
available 24/7 where you can ask him questions, whether it's Karpathy
all the way through to Richard Feynman.
And you can learn physics with them on the headphones while you are teaching that subject
or learning that subject.
And that's something that we are seeing pockets of.
Like a great example is Masterclass where Masterclass of course collaborates
with incredible teachers to deliver static lectures.
But recently they launched an interactive version of that.
So I don't know if that will be a good reference for this audience,
but we recently worked with them on bringing Gordon Ramsay that can teach you cooking.
So while you are in the kitchen he can shout at you effectively to get better.
Or maybe a better one is Chris Voss where you can of course learn negotiation,
but you can learn by negotiating with Chris live on the phone to get better,
which I thought was a phenomenal subject.
Having negotiated against Marty a number of times around financing rounds,
I understand now.
I think it helps you to say this, but I think the opposite is true.
I have some more questions.
I want to save time for the audience as well.
As Constantine mentioned, more than 100 million of net new ARR in Q1.
Obviously the business is going very well.
And you're sort of pioneering the startup founder building of foundation model applications.
Any counterintuitive lessons about building a company in this era that for the founders in the audience,
they may want to take home with them?
So we are, just for reference, we are just over 400 people, over 400 million in revenue,
but still keep the teams extremely small.
So it's rough, arbitrary, a little bit.
It's less than 10 people.
It's for each of the research product.
Even the go-to-market ops talent teams are all smaller than that size.
Most of the people will have 10 direct reports or so.
So it keeps it relatively flat and allows us to move a little bit quicker.
One thing that we've done, which is in this model, and very surprisingly, this is a very similar model that we've seen actually with the government of Ukraine.
Each of the teams, even the teams that aren't technical teams, will have engineers within them.
So our people team, our go-to-market team, our legal team will have an engineer in that team that helps to build, of course, automation, upscale, uplevel the rest of the people.
And recently, that really helped because, as I'm sure many of you are going through, everybody will be vibe coding and coding a lot of the help, even if they are not technical.
So now, that kind of shifted the responsibility, not responsibility, but shifted the requirement of how good the review needs to be for a lot of that work.
You have security, infrastructure implications.
You will want to make sure that the output is right.
And I think on the engineering side, you can put that expectation.
On the non-engineering side, the ability to do that is relatively hard.
So that technical research in those teams helped us a lot to figure this out.
And in general, there's just so many incredible work you can do by having that, whether that's the scraping on the hiring and recruiting front,
or analyzing what worked in the past to improve in the future, whether that's upskilling the legal team on how to use those tools,
and then figuring out ways of-- we recently introduced this scoring system.
For those on the go-to-market on the sales side, you frequently will end up in this negotiation with your sales team of,
can I give indemnity provisions?
What's the liability cap?
Can I give the set of clauses?
And then you kind of need to draw the line of how many things you give.
And I ended up being in so many of those conversations that we gave already a lot or we didn't.
So now we introduced the scoring system that you can give per size of the customer.
You can just give a few of those points out and in.
We just made it so much easier.
And of course, that's fully automated now with how we work across that team.
So that was one of the intuitive.
Small teams, bringing technical talent in the non-technical teams, keeping your activity flat.
We also have no titles, which allows us to bring people and really optimize for impact that they are having.
And then you can grow as quickly as you want.
The tenure will not define this.
And many more.
So we'll see.
It's a four-year-old company.
So we'll see if that helps.
Any questions?
Are you seeing people deploy voice agents to actually negotiate on their behalf?
And then are you starting to see agents actually negotiate with agents?
Sorry, I do three-part questions.
When that world happens, do you think the agents are actually talking to each other the way that humans talk to communicate and negotiate?
Or do you think it's beep-boop-beep-boop?
Do you think it's all done instantaneously?
How's that world going to look like?
So one, early inklings of that.
We haven't seen any truly successful on the negotiation front.
It was like more, you know, kind of order-taking.
What's the price?
Can we capture that?
And then kind of goes back to the team.
So not real negotiation.
But there's a few startups that we see, especially on any organizational shifts of,
can I organize this event?
Calling a lot of places.
Getting the price and then calling again with like our budget.
So that is happening and I think this will shift.
I think emotional intelligence will like, this is the big part that will start being important in a lot of that work.
Where it's not only the content that matters, but how you deliver, when you pause that work.
And then maybe the extreme version of that, which agents are not, like most of the people wouldn't do it.
And they are not good at that.
Today you will see a lot of interruptibility built in, where human can interrupt the agent.
But with negotiation you also want the opposite, where agent will interrupt the human.
It's kind of the extreme version of that.
On the second part, on the agent to agent part, some of you might have seen this.
We did a hackathon over a year and a half ago.
And that was exactly the case, where agent was speaking with another agent.
They detected that they are both agents.
And they swapped over to a different language.
And that was like a more efficient transmitter of information than just the classic spoken word.
And I think this will happen, 100%.
Like the big question will be really voice.
Will it be other transmission of information?
And depends truly on what the infrastructure is built for.
And I think this will define that experience.
Adios.
I'm curious how you're thinking about the need for voice in the future where agents do more and more of the work.
So basically what are the kind of use cases maybe where human conversation.
I think it's more of a follow up to the last question.
Like first, you all of us will have so many different devices around us.
And step from that you will have robots around us.
So of course, voice will be such an important interface to instruct and be able to interact with those devices.
In many ways, I feel like we see a lot of developments of intelligence.
But then the real bottleneck of the future will be how we communicate with that intelligence.
And I hope voice and visual part will be a big unlock to be able to actually get the most of that intelligence value in those settings, which isn't yet possible.
But on the flip side, the value of the human interaction will only increase.
So whether that's the events like this one, whether that's events with your favorite artist will increase in value with that ability of having voice all around you.
But the trust will be such a big part and something we optimize for in between the agent and human of, you know, in the future where all of you, all of us will have a voice agent, for example, to call and book a restaurant or give information to a healthcare appointment.
All of that will require such a high degree of trust that this is you and authenticated you.
So there will be like a level of encoding and decoding for real, then encoding and decoding for watermarked opted in human.
And then by default, everything else will be fake, which is kind of the opposite of how it is today.
You have detect for AI, but it will detect for real authenticated AI in the future and assume it's fake.
Andre spoke earlier about jagged intelligence.
Do you see similar odd places in audio where models are good and bad that you might not expect?
And yeah, what are they?
There's still so much on the bad side.
I think that, you know, like we spoke a little bit about where we see the voice agents working.
So like this combination of the models together and support settings works really well, works reliably.
And early sales starts working, but like the moment you start swapping to a true emotional interaction, not yet working.
It doesn't get the emotion that well.
It's slightly too slow.
So that is still like I think a big step change that should work.
Same will apply in a very different domain on the music side.
I think in the music side you can get good production music.
You cannot get top charts music even with artist input.
I think this will change over the next year or two.
Yeah, of course.
Andre's take was that the reason for that was that the labs were basically training for the stuff that had economic value.
Where you're training your models, is that true of you?
Are you basically training for the things that make the most money?
Or is it that there are some challenges that are genuinely harder than others?
We try to train the models, build a product and ecosystem that will deliver, of course, the biggest impact for all our customers, all users.
Which should correlate, of course, with the revenue in the long term.
So like that long term perspective, it's going to be like minimal in the next few years, so not next year.
So frequently we will train the models that might not provide that value in the short term.
Or even step before, we'll spend so much time labeling the data, not only the what of audio, but also how of audio.
Like what emotions did I use?
What is my voice described as?
What is this music described as?
So we assembled a team of now thousand plus people that have been voice coaches, musicians, artists before that can help us annotate that behind the scenes.
And that will not provide a value in the next six to 12 months, but we think well in the next 10 to 12 to 24.
And then you also need to collect that data, which frequently just isn't that accessible as well.
Last one, and then we'll go to lunch.
Hey.
Can you hear me?
Yes.
Big fan of yours in 11 labs.
Thank you.
So what do you think from the model air perspective, what do you think are the modes here with audio models?
The labs are going there, not going there.
What are the kind of, you know, in this sausage making of making a real good frontier audio model?
What are the main defensible parts there?
So, of course, we do a variety of models and recently had a pleasure of meeting Jensen and he was commending on a few of those models.
And he said that our speech to text or speech to text models are technology and text to speech is artistry and we are all artists.
So he gained a client for life.
But of course, we do believe there is a little bit of that to really fix text to speech and fix that emotionality.
You will need to be really focused on that space.
You really need to get in front of users, collect the data, collect the preferences, use that to fine tune the models.
And then there is a domain specificity in how you actually bring those models to production.
In healthcare, very different than in financial services, very different than in education or experiences.
So that's on the model layer.
I think there will be continuous advantage that if you actually care about the quality, that actually spending the time on the model work will help you keep that advantage.
But to your point, the models and like a lot of use cases will use a model as just a small part of their stack.
And that's where we spend a lot of time like beyond going beyond the research on the product side of how you understand a user's problem, the workflow that they need.
And voice agents is combining the audio models with knowledge and bringing that inside of the system.
How you bring it outside with telephony systems so you can interact across channels.
How you evaluate, test, and monitor.
And then as you create, whether that's in the agent space, whether that's in the creative space, the same understanding.
You build the ecosystem.
And that's what we hope to build across Eleven Labs.
A place where, whether it's distribution and brand that people can trust.
The platform where you have pre-existing set of work that you can start off.
Whether it's a template for creating an agent, template for creating a workflow in a creative space.
Or whether that's a voice.
And we had a pleasure now of having over 20,000 voices that people created, contributed, that you can use across language, styles, and voices.
And I think that will be an increasingly important layer of how you are able to cater to that diversity.
Make it easy for people to start and really understand that workflow.
All right.
I'm going to hand it back to Konstantin Mahdi.
Thank you.
Andrew, thanks for being partner.
Thank you guys.
Amazing.
Thank you guys.