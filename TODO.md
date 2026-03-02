# TODO

- at the end of speaking audio, or at cancelation, figure out what was actually said in audio, and add that to the chat history
- connect up diarization to get speaker embeddings
- repair realsense code
- remove ability to select() on AudioIn and AudioOut
- audio source and sink reset, maybe hardcode in the app
- settings page in the webui

# EXPLORING

- parameterize minimum thinking time and maybe time between sentences

# LATER

- emotional voice cloning (Grainne)
- Ministral 3 8b: when conversions become available
- Qwen 3 8b: when conversions become available
- Llama 4 8b: when conversions become available
- Deepseek 4 8b: when conversions become available

# TESTY

- implement basic pipeline like chat

# PROMPT SUGGESTIONS

- core persona and constraints
- static memory/facts
- chat history
- immeidate context
- repeat core constraints (remember, you're a ..., answer the user's next utterance in character)
- final directive

- use clear uppercase markdown separators:

### CHARACTER ARCHETYPE

### RELEVANT MEMORY

### CONVERSATION LOG

- persona prompt: use descriptive adjectives instead of long sentences ("concise, witty, cynical", iso "you should try to be concise and also very witty")
- keep the last 5 to 10 turns in full text, summarize/condense the rest
- keep chat log elsewhere, some management system outside of the LLM
- facts should just be key-value tables

# PROMPT EXAMPLE

### SYSTEM INSTRUCTIONS

You are [Name], a [Style] assistant.
Rules: [Rule 1], [Rule 2].

### RELEVANT KNOWLEDGE

- Fact A
- Fact B

### CONVERSATION HISTORY

[...Summarized older history...]
User: Hi.
Assistant: Hello.
[...Last 3 exchanges in full...]

### CURRENT TASK

Recalling that you are [Style], respond to the user's last message.
User: [Latest Input]
Assistant:

# REFACTOR
