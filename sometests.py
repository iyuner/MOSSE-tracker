bounding_boxes = []
with open("../Mini-OTB/anno/Basketball.txt", 'r') as f:
     for i in range(5):
        line = next(f).strip()
        # separate the numbers in the line and convert them to integers
        line = [int(x) for x in line.split(',')]
        bounding_boxes.append(line)
print(bounding_boxes)