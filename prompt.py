import sys

from gpt4all import GPT4All


PARAMS = dict(
    temp=0.9,
    top_k=1_000,
    top_p=0.7,
    n_batch=1024,
    max_tokens=1_000,
)


def do_setup():
    print('Loading...')
    model = GPT4All(
        'mistral-7b-instruct-v0.2.Q5_K_M.gguf',
        model_path='./',
        allow_download=False)
    return model


STOP_GENERATING = False


def end_turn(token_id, token_string):
    if STOP_GENERATING:
        return False

    return True


def print_response(response_it):
    for response in response_it:
        print(response, end='')
        sys.stdout.flush()

    print()


def read_until_eof():
    pending = []

    while True:
        data = sys.stdin.readline()
        if not data:
            break
        pending.append(data)

    return ''.join(pending)


def do_loop(model):
    print('Prompt for LLM (type ^D to finish input):')
    prompt = read_until_eof()
    print()
    print()
    print('> Received prompt:')
    print(prompt)
    print()
    print()
    print('> Generated response:')
    print()
    response_it = model.generate(
        prompt=prompt,
        streaming=True,
        callback=end_turn,
        **PARAMS)
    print_response(response_it)


def main():
    global STOP_GENERATING
    try:
        model = do_setup()
        do_loop(model)
    except KeyboardInterrupt:
        STOP_GENERATING = True
        print('Exiting...')

    sys.exit(0)


if __name__ == '__main__':
    main()
