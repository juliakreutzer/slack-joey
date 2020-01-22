# Talk to Joey: How to deploy a Joey NMT model as a slack translation bot

This is a quick guide on how to locally deploy a trained [Joey NMT](https://github.com/joeynmt/joeynmt) model as a [slack bot](https://slack.com/help/articles/115005265703-Create-a-bot-for-your-workspace). 
It's a great way to get a good feeling for what it has learned and a very simple way to show off your model without implementing a front-end.

Disclaimer: Not made for long-term or production-ready deployment, since we're not using a "proper" webservice.

## Requirements
- You need a trained Joey NMT model. See [here](https://github.com/joeynmt/joeynmt#training) for instructions on how to train one, or use one of the [pretrained models](https://github.com/joeynmt/joeynmt#pre-trained-models).
- In order to provide translations, Joey NMT needs to be running on a *machine*. GPUs are faster, but the reponse time of a model running on CPU should still be bearable (imagine a human typing - it's still faster than that ;)). You can only query translations as long as the job on the machine is running, so best would be a server. For short-time demos, your local machine should be fine, too.
- This code runs on Python3.6.
- We're using [`ngrok`](https://ngrok.com/) to expose a locally deployed model to the outside.

## Setup
### Installation
Install the required packages:

`python3.6 -m pip install -r requirements.txt` 

This includes Joey NMT (for CPU - install it manually for GPU, see next section).

### Joey NMT 
- You need to install Joey NMT and its requirements first, see [here](https://github.com/joeynmt/joeynmt#installation).
- Train a model. Let's assume it's stored in `my_model_dir`. This directory should contain at least one checkpoint, the vocabularies and the configuration file.

### 1. Create an app
- Create or choose a channel in your slack team to integrate your bot. This is where the bot reacts to every incoming message by anyone. Let's call this channel `BOT_CHANNEL`. Write it's name in `bot.channel`. 
- Create an app for your workspace. See this [tutorial](https://github.com/slackapi/python-slackclient/blob/master/tutorial/01-creating-the-slack-app.md).
    - Write the app's name (`BOT_NAME`) into `bot.name`. 
    - Define bot token scopes: `app_mentions:read`, `chat:write`, `incoming-webhook`, `channels:read` are needed. You need to re-install the app anytime you change the permissions.
    - Authorize the app for the workspace and assign it the new channel.
    - If this is successful, you'll receive a bot token. It should start with `xoxb`.
    - Write the bok token into `bot.token`.
    - Add the bot to the channel in the slack workspace.
    - Store the sign-in secret in `bot.signin`.
- We're going to use the [Event API](https://api.slack.com/events-api) to make the bot subscribe to events in slack. (The RTM API is no longer available for new apps.)
    
### 2. Subscribe to events
- Install [ngrok](https://ngrok.com/). It will allow us to expose a local service to the public.
- Start ngrok on a port 3000: `./ngrok http 3000`. In order to interact with your app, this process needs to be running.
- Copy the url that ngrok reports. It should look like `http://somerandomsymbols.ngrok.io`.
- Enable Event Subscriptions for your app, as described [here](https://api.slack.com/events-api#subscriptions). 
- For the Request URL, use the ngrok URL with a suffix: `http://somerandomsymbols.ngrok.io/slack/events`.
- Run `python3.6 main.py my_model_dir` to start the app. 
- After verification, subscribe to bot events: `app_mention`, `message.channels`, `message.im`.
- Click `Save changes` to save the changes for your app. 

### 3. Running 
- In slack, move to the `BOT_CHANNEL` and write a message. Your bot should automatically reply. 
- In addition to that, the bot reacts on mentions, so addressing `@BOT_NAME` will make it respond.

## Configurations and Customization
- Make sure to edit the `config.yaml` in the `model_dir` according to your use case. Mind the setting for the following:
    - `use_cuda`: set to False if running on CPU, True when on GPU.
    - `max_output_length`: sets the maximum output length
    - `beam_size`: beam size for decoding, 1 is greedy decoding.
- We assume the data is pre-processed with the MosesTokenizer if you set `tokenize`. If you want to use a different one, modify the code accordingly. 

### Interactions
See the docu on how to create bots for slack workspaces and explore the links there: https://slack.com/help/articles/115005265703-Create-a-bot-for-your-workspace
You could add more interaction modes, language id to activate different bots, etc. - please make a PR to this repo if you implement a cool extension :) Rhis is really just the bare bone.

