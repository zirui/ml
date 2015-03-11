#!/usr/bin/python
#encoding:utf8

import sys
import re
from collections import defaultdict 
from string import punctuation

from quan_ban_convert import  qun2ban 

reload(sys)
sys.setdefaultencoding('utf8')



pt = re.compile('"(.*)","(.*)"')



whitespace_pt = re.compile(ur"\s+")

#ch_punctuation_pt = re.compile(u'[、。，；]')
ch_punctuation ='、。，；'
#en_punctuation ='\'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
en_punctuation ='\'!"#$%&\'(*+,-./:;<=>?@[\\]^_`{|}~\''

desc_pt = re.compile(u'^(\(?[0-9]\)?)+([^\u578b])(.*)')

right_parenthesis_pt = re.compile(u'(.*)(\(.*\))')

#all_punctuation = '\'!"#$%&\'(*+,-./:;<=>?@[\\]^_`{，、|}~\''

all_punctuation = dict((ord(char), u'') for char in u'\'!"#$%&\'(*+,-./:;<=>?@[\\]^_`{，、|}~\'')


#

def getopts():
    if len(sys.argv) < 3:
        print 'invalid input'
        sys.exit()

    return sys.argv[1],sys.argv[2]



def clean_str(desc):
    #remove "
    desc = desc.strip('"')

    #convert to ban jiao from quanjiao
    desc = qun2ban(unicode(desc))

    #remove whitespace
    desc = desc.strip()

    #remove punctuation
    desc = desc.strip(en_punctuation)
    desc = desc.strip(ch_punctuation)

    
    desc = desc.strip()


    #remove prefix number of diag desc
    desc = desc_pt.sub(r'\2\3',unicode(desc))


    desc = desc.strip()

    #remove punctuation again
    desc = desc.strip(en_punctuation)
    desc = desc.strip(ch_punctuation)
    desc = desc.strip()

    #remove comment in right parenthesis
    desc = right_parenthesis_pt.sub(r'\1',desc)
    desc = desc.strip()



    #remove all puncutaion 
    #desc = desc.translate(all_punctuation)

    #remove '术后'
    desc = desc.replace('术后','')

    #remove whitespace
    desc = whitespace_pt.sub('',desc)
    #remove '\t'
    #desc = desc.replace('\t','') 


    return desc



def process(infile, rst_file_name):
    data = defaultdict(int)
    with open(infile,'r') as file:
        for line in file:
            #parts = line.strip().split(',')
            rst = re.search(pt, line)
            if rst is None:
                print line
                continue


            parts = rst.groups()
            if len(parts) != 2:
                print parts
                continue

            desc, cnt = parts 

            desc = clean_str(desc)
            if desc == '' or desc.isspace():
                continue
                    
            cnt = cnt.strip('"')
            try:
                data[desc] += int(cnt)
            except Exception as e:
                print cnt 
                print e
                continue
            


    with open(rst_file_name, 'w') as rst_file:
        for desc in data:
            cnt = data[desc]    
            line = '%s\t%s\n' % (desc, cnt)
            rst_file.write(line)


def test():
    str = '"3、原发性骨质疏松症"'
    #str = '"1、原发性骨质疏松症"'
    str = '"1.麻痹性外斜（右）"'
    str = '急性淋巴细胞白血病第9次化疗（B细胞性，中危）'
    str = ' 右乳癌? 1术后'
    print clean_str(str)


def main():
    #test()
    #return


    infile, rst_file = getopts()
    process(infile, rst_file)


if __name__ == '__main__':
    main()
