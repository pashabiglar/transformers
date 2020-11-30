from pylint.lint import Run

results = Run(['load_trained_model_print_attention_weights.py'], do_exit=False)
# `exit` is deprecated, use `do_exit` instead
print(results.linter.stats['global_note'])

# from pylint import epylint as lint
# (pylint_stdout, pylint_stderr) = lint.py_run('load_trained_model_print_attention_weights.py', return_std=True)
# print(pylint_stdout.linter.stats['global_note'])