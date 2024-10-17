from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom

raw_route_file = {
    'boston-seaport': './maps/boston-seaport/boston-seaport.rou.xml',
    'boston-thomaspark': './maps/boston-thomaspark/boston-thomaspark.rou.xml',
    'singapore-onenorth': './maps/singapore-onenorth/singapore-onenorth.rou.xml',
}
d2rl_route_file = {
    'boston-seaport': './maps/boston-seaport/boston-seaport-d2rl.rou.xml',
    'boston-thomaspark': './maps/boston-thomaspark/boston-thomaspark-d2rl.rou.xml',
    'singapore-onenorth': './maps/singapore-onenorth/singapore-onenorth-d2rl.rou.xml',
}

for city in raw_route_file:
    tree_raw = ET.parse(raw_route_file[city])
    tree_d2rl = ET.parse(d2rl_route_file[city])
    root_raw = tree_raw.getroot()
    root_d2rl = tree_d2rl.getroot()
    for route in root_d2rl.findall('route'):
        root_d2rl.remove(route)
    index = 0
    for vehicle in tqdm(root_raw.findall('vehicle')):
        for route in vehicle.findall('route'):
            new_route = ET.Element('route')
            new_route.set('edges', route.get('edges'))
            new_route.set('id', 'route_' + str(index))
            root_d2rl.append(new_route)
            index += 1
    # tree_d2rl.write(d2rl_route_file[city])
    xml_str = minidom.parseString(ET.tostring(root_d2rl)).toprettyxml(indent='    ')
    with open(d2rl_route_file[city], 'w') as f:
        f.write(xml_str)
