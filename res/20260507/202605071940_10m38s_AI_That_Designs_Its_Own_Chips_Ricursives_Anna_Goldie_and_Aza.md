---
title: AI That Designs Its Own Chips: Ricursive's Anna Goldie and Azalia Mirhoseini
uploader: Sequoia Capital
channel: Sequoia Capital
channel_url: https://www.youtube.com/channel/UCWrF0oN6unbXrWsTN7RctTw
duration: 638
upload_date: 20260506
webpage_url: https://www.youtube.com/watch?v=K05Dh-QjM8c
id: K05Dh-QjM8c
categories:
  - Science & Technology
---

# AI That Designs Its Own Chips: Ricursive's Anna Goldie and Azalia Mirhoseini

One of the themes that we've heard throughout the day is that neural nets are replacing a lot of
traditional tools and I think one of the most exciting application categories where we've
actually seen that come to life is within chip design where neural nets are now becoming superhuman
at certain parts of the semiconductor design process and so I'm thrilled to introduce Anna
and Azalea they were the co-creators of alpha chip which did exactly this at Google and was
used on multiple generations of TPU and I've now started a company to build this thank you and
welcome Anna and Azalea hi everyone Azalea and I are so excited to be here today to talk about our
new company recursive intelligence where we're doing AI for chip design and chip design for AI
we've been working together closely for the last 10 years across Google brain anthropic
deep mind and then because it wasn't enough to work at one institution in parallel I started my PhD while
continuing to work full-time and Azalea joined the Stanford faculty so you know we have been working
in many different places together for a long time but our thesis in this company is that chips are the
fuel for AI and that we should be using AI to design to optimize and automate the chip design process and
close this recursive self-improving loop between AI and its physical substrate we started on this direction in 2018 with our work on alpha chip
where we developed a deep reinforcement learning agent that was capable of generating superhuman chip layouts
this work was published in nature but the interesting part about it in our opinion was it was actually used in the tape out of real chips
so the last four generations now of Google's AI accelerator chips TPU data centers CPUs called axion pixel phones and also autonomous vehicle chips and in addition to adoption by external
companies like media tech and so we decided to start this company to take this work to the next level and take on all of the chip design work flow
we see the company in three phases so currently we're in phase one where we want to accelerate the chip design process so today there are two long poles one is physical design and which is placing the billions of standard cells or billions of transistors and routing billions of these components onto a chip canvas and design verification which is verifying the correctness of the
the logic of the logic of that chip each of these can take up to a year and involves hundreds or thousands of human experts and the stakes are extremely high so we've heard estimates like one day of delay of an nvidia chip cost like a blackwell cost the company something like 225 million dollars in lost opportunity cost so you want to help existing chip makers get to market faster build faster cheaper and more environmentally friendly chips but i think in phase two of the company we want to democratize chip design
so you want to become a platform for designing new hardware where we can take as input like a workload see like the next cloud model design an architecture that massively accelerates that workload and then do the entire design process all the way to gds to clean which is the format that we send to the fabs for manufacturing
and in that case we can and in that case we can massively unlock the number of customers that we could serve like any company that has a workload that that they serve at sufficient scale could benefit from custom chips even if they don't have teams of hundreds or thousands of human experts
uh and then phase three of the company would be vertical integration so if we have this capability to quickly design highly performant chips why not build our own chips why not train our own models and co-evolved
them and serve intelligence at a price or a capability that would be impossible to match
so azalia is going to talk a bit more about our approach in this company yes if i can stop on this slide uh so on the right side we are showing you the flow the traditional flow for chip design it starts with architecture design and it goes all the way to sign off and that's what you send to the fabs
and as you can see there are many components here and the way these uh kind of phases are done today with human experts in the loop working with commercial tools that sometimes takes days to run for a single iteration of an optimization so our approach here at recursive is to
to first redesign the way these tools uh perform make them a hundred thousand x faster and then they're primed to be used with ai because as you know our ais really like fast iteration loops and they can just exponentially learn more and co-optimize across a very very large space if we enable them to do so
so by co-designing across the stack we uh what this enables is unlocking massive performance improvements and time to market which comes from both the co-design and the automation
um to just show you a glimpse into what we are building here is an sp a static timing analysis engine this is one of the uh a very challenging component of physical design and what we are showing here is that we are we have built a tool that correlates
with the commercial tools uh very high fidelity and what we can do here is do so a thousand x faster now imagine if you are doing an ai tool use or an rl loop we can we now have this uh kind of feed feedback signal that we can use in the optimization we can do a lot more with it um here is a example of how our outer outer loop our ai works with this tool like here early on
is what we can do a single iteration of our inner loop but as the ai optimizes around the recipes that are possible to use these tools with we can get significantly more performance
so taking a step back uh what recursive uh is enabling is a new era there which we call designless just like fabless um enabled by companies like tsmc
uh made made made made made it such that we can have nvidia apple other companies focus on designing chips and send off the designs for fabrication elsewhere we want to be uh the platform for uh chip design so companies can focus on the application modeling and other layers and we can be the compute and hardware that enable those applications um and the impact would be that we we can democratize chip design and enable a lot more
a lot more variety and performance types of chips possible so right now um there are a few mainstream chips for ai inference but as you can imagine and as a lot of talks and conversations today allude to we are going to need a whole lot more performance in the coming years and one way to unlock a lot of performance is through customization so when we build chips that are truly customized to the workloads that we are serving
we are serving.
And we at Recursive want to be the platform that
enables this Cambrian explosion of chips,
so we can build a lot more variety of chips that
are really custom to the types of workloads
that the companies and the users care about.
And you can imagine these chips can
enable a very large workload like a frontier model,
or they can enable a very low power or high throughput
or other kind of variations of performance
that we would need.
And finally, we have an amazing team.
Our company is a little unusual in that we
have this subset of people who are very expert in LLM.
They have worked on Cod, Gemini, Grog, and such in the past.
And now we have put them together with these experts
in chip design.
And so it's a very great mix.
And we are very glad that we get to build together.
Yeah.
And with that, we can conclude the talk.
I'm happy to answer your question.
Can I work on the shape of the chip placements that you end up
seeing out of these models?
Oh, shoot.
We should have shown some of the layouts.
So I think just like an alpha chip, we're seeing these kind of curved,
organic-looking shapes to our placements.
So human experts, they tend to make these very aligned, regular-looking placements.
But the AI-generated ones look more organic, curved, which minimizes wire lanes, improves performance.
But it was kind of shocking to physical design engineers when they first saw them.
A question on the cost of the specialized chips.
I totally understand that you could make a better chip with AI and have better placements and stuff.
But I think Azalea was also making the point that you could make specialized chips.
How does the scale work out?
I don't know the first principles of chip design.
Can you make thousands of different chips and make them as cheaply as one Hopper architecture?
I mean, there's-- yeah, go ahead.
Yes, there is definitely what we are doing here is like we are going into a new regime
where we have-- we can work compute to our advantage.
So by scaling compute, we can reduce the run times to design the chip and also to enable more performance chips.
So we are basically introducing a knob.
Now, through customization, given the scale at which AI workloads are going to be run,
the economy of skills is going to-- is working itself, right?
Even a 1% improvement in a chip that serves a frontier model is a massive gain and success if you could enable that.
But at different scales, we have different performance gains.
And again, what we are talking here is that we are using compute
to bring automations and better performance, and that's the knob that we can play with.