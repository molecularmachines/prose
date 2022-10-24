from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("results_dir", None, "Path to results directory")
flags.mark_flag_as_required("results_dir")


def main(argv):
    del argv


if __name__ == "__main__":
    app.run(main)
