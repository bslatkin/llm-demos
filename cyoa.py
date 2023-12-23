import sys
import threading

from gpt4all import GPT4All


def do_setup():
    print('Loading...')
    model = GPT4All(
        'mistral-7b-instruct-v0.1.Q4_0.gguf',
        # 'nous-hermes-llama2-13b.Q4_0.gguf',
        model_path='./',
        allow_download=False)

    setting = input('Provide the setting for the story: ')

    print()

    system_prompt = f"""\
You are an adventure text game. Describe each decision point to the player. Output one sentence at each step. Always wait for the player to decide what to do next. Never end the story. Never break character. The setting for the story is: {setting}
"""

    print(system_prompt)

    return system_prompt, model


STOP_GENERATING = False


def end_turn(token_id, token_string):
    if STOP_GENERATING:
        return False

    return '#' not in token_string


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
            prompt='### Game:\n',
            temp=0,
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
                prompt=f"### Player's command:\n{prompt}\n\n### Game:\n",
                temp=0.5,
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
