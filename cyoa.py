import sys

from gpt4all import GPT4All


def do_setup():
    print('Loading...')
    model = GPT4All(
        'orca-2-13b.Q4_0.gguf',
        model_path='./',
        allow_download=False)

    setting = input('Provide a setting for a "choose your own adventure" story: ')

    print()

    system_prompt = f"""\
You are an adventure text game. You will briefly describe each scene to the player. You will always provide the player with multiple options to choose from for their next action in each scene. The setting for the story is:

{setting}
"""

    return system_prompt, model


STOP_GENERATING = False


def end_turn(token_id, token_string):
    if STOP_GENERATING:
        return False

    if '>' in token_string:
        return False

    return True


def print_response(response_it):
    for response in response_it:
        print(response, end='')
        sys.stdout.flush()

    print()


def do_loop(system_prompt, model):
    with model.chat_session(
            system_prompt=system_prompt,
            prompt_template='{0}'):

        response_it = model.generate(
            prompt='> GAME:\n',
            temp=0,
            max_tokens=10_000,
            streaming=True,
            callback=end_turn)
        print_response(response_it)

        while True:
            print()
            prompt = input('> ')
            if prompt == 'quit':
                break
            if not prompt:
                continue

            print()
            response_it = model.generate(
                prompt=f"> PLAYER'S COMMAND:\n{prompt}\n\n> GAME:\n",
                temp=0,
                max_tokens=10_000,
                streaming=True,
                callback=end_turn)
            print_response(response_it)


def main():
    global STOP_GENERATING
    try:
        system_prompt, model = do_setup()
        do_loop(system_prompt, model)
    except KeyboardInterrupt:
        STOP_GENERATING = True
        print('Exiting...')

    sys.exit(0)


if __name__ == '__main__':
    main()
