text = open('test.txt', 'r').read()
rows = text.split('\n')
file = open('test2.txt', 'w')
for row in rows:
    sentence_with_bad = row.split('|||')[0]
    file.write(sentence_with_bad + '\n')
