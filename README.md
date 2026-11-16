# GLM

A small character-level language model I'm training with
[nnet](https://github.com/EliLevasseur/neural-network), the C++ neural network
framework I'm writing from scratch. There's no Python and no outside ML library
here. The tokenizer, batching, training loop and text generation live in this
repo, and all the math runs through nnet.

For now it's a simple model: it reads the last 32 characters and predicts the
next one. I'm training it on TinyStories, and using it to learn how language
models work from the bottom up before I build a transformer.

## The data

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is a
dataset of short children's stories written with a deliberately small
vocabulary, These imple, repetitive stories are a much better fit for
a model this size than a novel.

Download the data with the command **!!! NEEDED TO BUILD !!!**

```bash
curl -L -o data/tinystories.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

The first 90% of the file is for training, which is about 20 million sliding windows,
and the last 10% is held out for validation.

## How it works

Text is read as raw bytes. The vocabulary is the 256 possible byte values,
with IDs from 0 to 255; no word dictionary is needed. A training window contains
32 IDs, and its target is the byte immediately after it. Windows use positions
in stored token sequences rather than separate copies of every overlapping window,
so the 22.5 MB file doesn't turn into gigabytes of copies.

The model is

```text
Embedding(256, 16) -> Dense(512, 128) -> ReLU -> Dense(128, 64) -> ReLU -> Dense(64, 256)
```

trained with softmax cross-entropy and Adam. The embedding gives every byte its
own learned row of 16 numbers, and the same table is used for all 32 positions.
Every epoch is checked against a fixed sample of 20,000 validation windows to find the error.
Training stops once validation hasn't improved for 3 epochs, and the best epoch
is the one that gets saved.

To write text it takes a prompt, predicts how likely each next byte is, samples
one with a temperature of 0.6, adds it to the end, and slides the window forward.

## Results (v1)

v1 is the model above, with two hidden layers and about 95,000 parameters. Its
best validation loss on TinyStories is 1.125 per character. With one hidden
layer it got to 1.20 on the same data, even with slightly more parameters
(about 103,000). Before TinyStories I trained on Alice's Adventures in
Wonderland alone, which only got to 1.78 because one book isn't enough text.

My goal for v1 was that most of 10 samples from "Once upon a time, there was a
little" continue with real words and a short grammatical sentence and all 10 did,
Here are three samples from the v1 checkpoint, each cut off at the end of a sentence:

```text
Once upon a time, there was a little bird named Amy. He looked again. She saw a big sad.
Once upon a time, there was a little boy named Tim. He loved to climb it too. They saw she was sad and play.
Once upon a time, there was a little boy named Tim. Tim was sad and her everyone were tree.
```

### What it can't do yet

Every sample falls apart in its second sentence. The model has the opening
down, but after it the model strings together common pieces like "was sad"
without making them into a sentence.
It also loses track of who it's talking about: Amy turns into "He" and then
"She" while her name is still inside its 32-character window, so that's
something the model hasn't learned yet, not something it can't see.

The window is the other limit. Anything more than about six words back is
invisible to it, so it can't stay with one character or a plot for a whole
story, and a question with its answer wouldn't fit. Letting this kind of model
see further means a bigger first layer that learns every position separately,
which is why a transformer comes next.

## Building

```bash
git clone https://github.com/EliLevasseur/glm.git
cd glm
cmake -S . -B build
cmake --build build
```

The build uses `-march=native` by default, which almost doubles training speed
but ties the programs to your CPU. Turn it off with the flag `-DGLM_NATIVE=OFF`.

## Running it

Training methodology for people who want to use this model:

```bash
./build/glm                      # trains on data/tinystories.txt
./build/glm data/alice.txt       # or any text files, one after another
```

Each run starts a fresh model and saves its best epoch as the next numbered file
in `checkpoints/`. When you give it several books, it prints the validation loss
on every book from that run after each one, so you can see how much it forgets
the earlier books.

To write without training, download my v1 checkpoint from the
[v1.0.0 release](https://github.com/EliLevasseur/glm/releases/tag/v1.0.0).
`glm_write` looks for numbered checkpoints in `checkpoints/` inside the folder
you run it from, so run it from the repo folder, eityher download it from the relase or via terminal then test the models writing skills!:

```bash
curl -L --create-dirs -o checkpoints/2.ckpt https://github.com/EliLevasseur/glm/releases/download/v1.0.0/2.ckpt
./build/glm_write 2 "Once upon a time, there was a little"
```

Write with the newest checkpoint, or pick one by number:

```bash
./build/glm_write "Once upon a time, there was a little girl"
./build/glm_write 2 "Once upon a time, there was a little girl" 300   # 300 = how many characters to write
```

Prompts need at least 32 characters since that's the window size. The window
size, layer sizes, learning rate, batch size, temperature and file paths are all
changeable in `include/config.h`.

On my Ryzen 9950X an epoch over TinyStories takes about 11 minutes. It runs on
one core: nnet only splits a matrix multiply across threads once the matrices
get big, and this model's are still small.

## License

MIT, see [LICENSE](LICENSE).
