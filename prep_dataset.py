import json
import os

sents="""గుజరాత్ కాంగ్రెస్ సర్దార్ సందేశ్ యాత్ర ముగిసింది
ప్రపంచంలోనే అతిపెద్ద టెలిస్కోప్ కానరీ దీవుల్లో ఉంది.
ఈ కొత్త టెక్నాలజీతో దక్షిణ ఢిల్లీలో చెత్తను సేకరించనున్నారు
గతం గురించి పశ్చాత్తాపపడటం మనకు అలవాటు.
మార్పు ఎప్పుడూ సులభం కాదు.""".split("\n")

# print(marathi_sents)
data=[]

for root, dirs, files in os.walk("./recordings/"):
    # print(root, dirs, files)
    for file in files:
        if file.endswith(".ipynb"):
            continue
        else:
            # Create the full file path
            full_path = os.path.join(root, file)
            wav_paths = []
            print(full_path)
            # for root, dirs, f in os.walk(full_path):
            #     # print(f)
            #     for tmp in f:
                    # Check if the file ends with .wav
            if full_path.endswith(".wav"):
                print("hi")
                # Create the full file path
                wav_paths.append(full_path)
                wav_paths=sorted(wav_paths)     
                print(sorted(wav_paths))

                for i in wav_paths:
                    if "id" not in i[:-4].split("_")[-1]:
                        val=int(i[:-4].split("_")[-1])
                    else:
                        val=int(i[:-4].split("_")[-2])
                    data.append(
                        {
                            "audio": i, 
                            "text": sents[val-1]
                        }
                    )

file_path = "telugu_data.json"

# Dump dictionary into the JSON file
with open(file_path, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)  # `indent=4` makes it more readable (pretty-printed)
