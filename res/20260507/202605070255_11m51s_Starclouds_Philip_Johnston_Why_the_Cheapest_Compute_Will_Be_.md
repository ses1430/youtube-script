---
title: Starcloud's Philip Johnston: Why the Cheapest Compute Will Be in Space
uploader: Sequoia Capital
channel: Sequoia Capital
channel_url: https://www.youtube.com/channel/UCWrF0oN6unbXrWsTN7RctTw
duration: 711
upload_date: 20260506
webpage_url: https://www.youtube.com/watch?v=94b6i5jI1nE
id: 94b6i5jI1nE
categories:
  - Science & Technology
---

# Starcloud's Philip Johnston: Why the Cheapest Compute Will Be in Space

thanks so much for having me my name is Philip Johnston and I'm the co-founder and CEO of Star
Cloud and just like the previous company we have also been abusing GPUs in ways they were not
designed for and so yeah we're building data centers in space mainly for the energy that we
can draw and I will spend the next five minutes explaining why it will soon make much more sense
to build data centers in space than it does to build them on earth and then I'll take five
minutes for questions so please start thinking of some questions before I do that I want to show a
quick video which is actually the deployment of star cloud one and this was it had five NVIDIA GPUs on it
but the most significant was the NVIDIA H100 chip and I'll just quickly play the video first
star cloud one separation confirmed so we don't you don't normally get as
great a deployment video by the way half the time it like deploys into the shadow so the reason this
was so significant is until this point many people thought you actually couldn't run state-of-the-art
terrestrial data center grade GPUs in space for two main reasons one is the thermal dissipation so
they're very power dense they produce a lot of heat and the second is the radiation tolerance so
people thought that you would have bit flips at too higher rate and so by with this chip we were the
first to train a model in space we actually trained nano GPT from Andre Carpathy and then we also were the
first to run a version of Gemini the first to do high powered inference on SAR data so other satellite data
and so it was a very significant step in proving that we can actually run the state of the art terrestrially
but the
yeah I think maybe to make the case for why it will soon make more sense in terms of energy cost
I'd like to quickly draw a comparison with with a solar project on earth since solar is the cheapest form of
energy that we have on earth so if you want to build a solar project to power a new data center
you have three main costs so the first is the cost of permitted land and in fact in north america
that's actually the largest cost or can be for most new solar projects the second is the cost of battery
storage and backup power because we're only you know we only have peak power for about four hours of the day
so we need to charge those batteries to use at night and then the last is the cost of the solar cells
themselves so how does that compare to building a similarly sized solar project in space well in space
number one we don't need to pay for permitted land so your biggest cost is gone you don't need to pay for
battery storage and backup power because we're 24/7 in the sun so your second biggest cost is gone
and then you need eight times less solar cells because one square meter of solar panel in space
produces eight times the energy of one square meter solar panel on earth so the only additional cost or
the main additional cost we have in space is the launch cost and so you can clearly see there's a
break-even point where the launch cost comes below the cost of permitted land batteries and solar and we
see that break-even cost to be around 500 a kilo so about 10x reduction from where we are today but that's
well within range of the launch vehicles that are coming online so for comparison starship is designed
to produce launch costs of around 10 to 20 a kilo and so i think i'll just finish by playing you a
one final concept video and this shows a constellation that we're building now so we've just filed with the
fcc for a constellation of 88 000 satellites each one's about 200 kilowatts it will enable us to deploy
on the order of 20 gigawatts of new compute capacity really just scratching the surface um with this new
constellation um and it will enable it's basically for all inference workloads and so this could be and yeah
maybe i'll start the video and you can get a sense of it so in this case it's to generate a 3d video but it
could also be for back office business processing agents code generation agents um they'll come up via
optical link to this constellation in this dawn dusk sun synchronous orbit means it's always in the sun 24 7
power sub 50 millisecond latency to anywhere on earth um all optically linked and this really is the start of
the largest infrastructure project ever i mean we're talking about just for this constellation of 88 000
we're talking about 100 billion dollars of capex spend which is actually much lower than it would
cost to do to do terrestrially um and not only is it the start of the largest infrastructure project it's
also in my opinion the start of a college of type 2 uh dyson sphere type civilization and potentially
college of type 3. um i will finish there and we have about four minutes for questions so any questions yeah
we'll start at the front uh the intuition on the availability uh the intuition on the availability of
solar is obvious yeah um can you just give us the napkin math on the the radiator equation
again yes so like dissipating heat for anyone that's thought about it it always feels hard and then i would
also please say something about the availability of dawn dusk that orbit is finite right yes yeah yeah
it's a great question so because space is a vacuum it's actually much harder you know space is only three
degrees kelvin so very low ambient temperature but because it's a vacuum um as you so rightly point
out it's actually quite difficult to dissipate that heat and what it requires is a large surface area
so that you can emit that in infrared so everything that's warm is glowing in infrared all the time if
you had an infrared camera on my face you see that i'm glowing and so the rough math on the surface area is
your your solar panels generate around 200 watts per square meter and the radiator if you keep it around 50
uh will dissipate around 800 watts per square meter so what that means is if you've got a um you need
about a quarter again the surface area in radiator than you have on um on solar panels so if you had
a 400 square meter solar panel you'd need an additional 100 square meters of radiator to dissipate that heat
um there's a very nice equation called the Stefan Boltzmann equation which basically says that the
rate that the the thermal dissipation is proportional to this to the fourth power of the temperature
so if you can jack up that temperature instead of being 50 degrees to 80 degrees which is like a 10
increase in kelvin then you can actually half the surface area of your radiator or close to half the
surface area radiator and so that's what we're working on with nvidia now if anybody was at gtc you'll
have seen jensen walk out to this um to the deployment video of star cloud one and then he spent uh five
minutes talking about the um the new space ruben one chip that we're working on and it's designed to run
at a hotter temperature without having a higher failure rate and the reason you want it to run a
hotter temperature is so that you can lower the mass on the radiator great question there yeah yeah
yeah it's a great question it's also related to the question um asked about the space in space um
i think so it's something we take incredibly seriously you know everybody needs to be a
responsible user of space we do and everybody else we're you know keen to make sure that space is
usable forever um for the first few satellites so you can solve it in a few ways if you fly at a
relatively low altitude the chance of a kessler type effect is is extremely low so um our first satellite
we're flying around 400 kilometers altitude that means that it will naturally do orbit within a few months
and so if you were to have a collision at that altitude by the time it gets around to the next
orbit you're already a few hundred meters below where you had the collision and the chance yeah chance
kessler is very very low um yeah as you fly higher it's actually extremely unpopulated those high
orbits because then you start to edge into the the van allen radiation belt um but i mean we actually have a
pretty good case study for this and that is um spacex is now operating around 10 000 satellites without ever
having a single collision um in in low earth orbit and the way that you do that is by having a pretty
sophisticated collision avoidance um the the other reason i think people think this is more of an issue
than it than it actually is and and the reason the space is so much larger than it looks is when you
see a map of all of those satellites each dot on those maps is about the width of california and you're
representing something that might be this wide by something the width of california and so people can often think
the space is very congested it's actually you we can easily fit on the order of uh terawatts of compute
in just this dawn-dust synchronous orbit without having um you know huge problems with collision avoidance
any other questions yeah uh is radiation like bit flipping is that something you actually have to think
about or consider how does that impact stuff yes it is something i have to think about um so the way that
we're solving it is just an enormous amount of ground testing so we've done a four rounds of testing at the
cyclotron down in knoxville um it's a high velocity proton particle accelerator um and we take all that
telemetry and then that informs our choice on shielding and then for heavy ions we have to go to the
brookhaven national lab and we basically run all the chips through um through the the space environment so
over a 24-hour period you can put it through five years worth of radiation dose um and then we take
all that data and we then use that to inform shielding but also software development choices for it
yeah yeah uh this is almost exclusively i mean actually for the foreseeable future will just be for
inference um and the reason the reason it's for inferences number one inference is going to be like
99 of the compute market very soon anyway so even if we you know we wouldn't want to right be running a
large training set um well running large training sets will be a very small percentage of the total in
five to ten year time of ai workloads but secondly it's very hard we would need a yeah we would need
to dock together a large five gigawatt um kind of structure then i actually have a video of that here
but i won't waste everyone's time with it unless anybody wants to see a video of a five gigawatt data
center in space do you guys want to see that all right we we made this video because we didn't want
people to be like oh you could never train a model in space so this is what a five gig or
four kilometer by four kilometer structure in space would look like um so this would be a starship
launch vehicle with a 40 megawatt um is what you can fit per starship launch vehicle um which will
connect to a central spine um which is connected to this enormous solar panel on the back there we have
a one kilometer by four kilometer radiator um yeah that that's how you would train a large model but
as i say it'll probably be at least 15 years before we get to anything uh like that one more i think
we're okay 40 seconds or did we yeah okay oh sorry um by when do you think the majority of the data
center will be in the space oh that's a great question and i actually wanted to ask all of you
let's run a poll so the the the question i want to ask is when do you think it will be cheaper to run
compute in space for anybody it could be for spacex or for us and the four answers will be um the next five
in five to ten years it'll be cheaper uh sometime after 10 years or never okay so who thinks within
five years it will be cheaper to run compute in space than terrestrially interesting who thinks five to
10 years who thinks beyond 10 years and who thinks never brave okay that's that's an interesting uh for me to see
um i don't yeah i think that's uh we are out of time so i will leave it there thank you very much for your time