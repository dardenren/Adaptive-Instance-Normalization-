lr=""
epochs=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --lr)
            lr="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

source ./venv/bin/activate

cmd="python3 main.py \
    --lr \"$lr\" \
    --epochs \"$epochs\" "

eval "$cmd"