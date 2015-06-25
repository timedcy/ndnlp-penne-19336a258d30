import inspect
import linecache
import traceback

debug = False
trace = False

def extract_stack():
    """Extract just the information needed from the current stack to print
    it later if needed."""

    framelist = []
    f = inspect.currentframe()
    i = 0
    while f:
        if i >= 3: # hack to get up to the line where the Expression was created
            framelist.append((f.f_code.co_filename, 
                              f.f_lineno,
                              f.f_code.co_name,
                              f.f_globals))
        f = f.f_back
        i += 1
    return framelist

def format_list(framelist):
    """Format the stack as returned by extract_stack."""

    new_framelist = []
    for (filename, lineno, function, globals) in framelist:
        linecache.checkcache(filename)
        text = linecache.getline(filename, lineno, globals)
        new_framelist.append((filename, lineno, function, text))
    return traceback.format_list(new_framelist)
