TEMPLATE_PATH=$PROJECT_PATH"../../data/templates.json"
BASE_PATH=$PROJECT_PATH"../../data/instruments/"
PARAPHRASED_PATH=$PROJECT_PATH"../../data/paraphrased-prompts/"

pattern="ACI.json"
for FILE in $BASE_PATH$pattern; do
  echo $FILE;
  bare=$(basename -- "$FILE");
  echo $bare;
  python make_prompts.py -f $TEMPLATE_PATH -p "$BASE_PATH$bare" -d "$PARAPHRASED_PATH$bare";
done
