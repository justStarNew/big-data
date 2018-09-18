from lorem import text
from collections import Counter
import operator

c = Counter(filter(None,text().strip().replace('.','').replace('\n',' ').lower().split(' ')))
result = dict(sorted(c.most_common(),key=operator.itemgetter(1),reverse=True))
result