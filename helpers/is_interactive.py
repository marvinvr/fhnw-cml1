def is_interactive(main):
    res = not hasattr(main, '__file__')
    if res:
        print('Running previous notebooks...')
    return res
