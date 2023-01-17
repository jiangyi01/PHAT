import json

def get_json(jobId, status, result="None"):
    formdata = {
        "jobId": jobId,
        "status": status,
        "result": result
    }
    return formdata

if __name__ == '__main__':
    # some testFile
    filename = './data.json'
    with open(filename, 'r') as file:
        # content = file.read()
        content = json.load(file)

    # content = content["points"]
    print(content)
    PATH = './dataprocess.json'
    with open(PATH, 'w', encoding='utf8') as f:
        # content.encode("utf-8").decode("unicode_escape")
        # f.write(content)
        json.dump(content, f, ensure_ascii=True)

    # str = "\u5317\u4eac\u5e02\u533b\u7597\u4fdd\u969c\u5c40\u5173\u4e8e\u4fee\u8ba2\u300a\u5317\u4eac\u5e02\u533b\u7597\u4fdd\u969c\u884c\u653f\u5904\u7f5a\u81ea\u7531\u88c1\u91cf\u57fa\u51c6\u300b\u7684\u901a\u77e5"
    # print(str.encode("utf-8").decode("unicode_escape"))
    # print(str)