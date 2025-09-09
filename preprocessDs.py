from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json, os, sys
from functools import partial

dataDir = 'enwiki_namespace_0'
outputDir = 'enwiki_formatted'
maxFiles = 100 # Testing only, set to >37 for all files for enwiki ds DS
maxLinesPerOutputFile = 10000
maxWorkers = 10              
minLineLength = 100

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_input_files(dirpath):
    try:
        names = sorted(os.listdir(dirpath))
    except FileNotFoundError:
        print(f"Input directory not found: {dirpath}", file=sys.stderr)
        return []
    return [n for n in names if not n.startswith('.')]

def blocks(f, size=65536):
    while True:
        b = f.read(size)
        if not b:
            break
        yield b

def get_file_line_count(path):
    with open(path, "r", encoding='utf-8') as f:
        return sum(bl.count("\n") for bl in blocks(f))

def getText(section, maxDepth=10, depth=0):
    results = []

    if isinstance(section, dict):
        keys = section.keys()
        if "sections" in keys:
            for sec in section["sections"]:
                if "has_parts" in sec:
                    results.extend(getText(sec["has_parts"], maxDepth, depth=depth+1))
                results.extend(getText(sec, maxDepth, depth=depth+1))

        if "type" in keys and section["type"] == "paragraph":
            val = section.get("value")
            if isinstance(val, str) and val[-1] == '.':
                results.append(val)

    elif isinstance(section, list):
        for s in section:
            results.extend(getText(s, maxDepth, depth=depth+1))

    return results


def process_line(line):
    try:
        obj = json.loads(line)
    except Exception:
        return []

    try:
        texts = getText(obj)
        return texts
        # return [t.replace("\r", " ").replace("\n", " ").strip() for t in texts if isinstance(t, str) and t.strip()] # This messes up some newline chars
    except Exception:
        return []


def process_file(inputDir, filename, outputDir, linesPerSplit, numWorkers):
    ensure_dir(outputDir)

    inputPath = os.path.join(inputDir, filename)
    if not os.path.isfile(inputPath):
        print(f"Skipping non-file: {inputPath}", file=sys.stderr)
        return

    try:
        totalLines = get_file_line_count(inputPath)
    except Exception as e:
        print(f"Can't count lines in {inputPath}: {e}", file=sys.stderr)
        totalLines = None

    splitIndex = 0
    writtenInSplit = 0
    baseName = os.path.splitext(filename)[0]
    def make_out_path(idx):
        return os.path.join(outputDir, f"{baseName}_split_{idx}.txt")

    outputPath = make_out_path(splitIndex)
    outputFile = open(outputPath, "w", encoding="utf-8")

    workers = max(1, min(numWorkers, (os.cpu_count() or 1)))
    mapChunksize = 64

    try:
        with open(inputPath, "r", encoding="utf-8") as fin, \
             ProcessPoolExecutor(max_workers=workers) as ex, \
             tqdm(total=totalLines, unit="line", desc=f"Processing {filename}") as pbar:

            for texts in ex.map(process_line, fin, chunksize=mapChunksize):
                pbar.update(1)
                if not texts:
                    continue

                for t in texts:
                    if writtenInSplit >= linesPerSplit:
                        outputFile.close()
                        splitIndex += 1
                        writtenInSplit = 0
                        outputPath = make_out_path(splitIndex)
                        outputFile = open(outputPath, "w", encoding="utf-8")

                    outputFile.write(t + "\n")
                    writtenInSplit += 1

    finally:
        try:
            outputFile.close()
        except Exception:
            pass

def main():
    ensure_dir(outputDir)

    fileNames = list_input_files(dataDir)
    if not fileNames:
        print(f"No input files found in: {dataDir}")
        return

    # Want to ensure decent line length, this filters out single dates
    targets = fileNames[:maxFiles]

    print("Files to process:")
    for file in targets: print(file)
    print("\n")
    for fname in targets:
        process_file(
            inputDir=dataDir,
            filename=fname,
            outputDir=outputDir,
            linesPerSplit=maxLinesPerOutputFile,
            numWorkers=maxWorkers
        )

if __name__ == "__main__":
    main()
