import sys

from gpt4all import GPT4All


PARAMS = dict(
    temp=0.9,
    top_k=100,
    top_p=0.6,
)


def do_setup():
    print('Loading...')
    model = GPT4All(
        'orca-2-13b.Q4_0.gguf',
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


def do_loop(model):
    print('Prompt for LLM (type ^D to finish input):')
    prompt = sys.stdin.read()
    print()
    print('Response:')
    print()
    response_it = model.generate(
        prompt=prompt,
        max_tokens=100_000,
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
