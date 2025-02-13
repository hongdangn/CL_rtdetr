
import os
import clip
import torch

labels_dict = {
    'person': 'A human being, usually depicted standing, walking, or engaging in various activities.',
    'bicycle': 'A two-wheeled, human-powered vehicle with pedals, handlebars, and a seat.',
    'car': 'A four-wheeled motor vehicle used for transporting passengers.',
    'motorcycle': 'A two-wheeled motor vehicle with a seat for the rider and, often, a passenger.',
    'airplane': 'A powered flying vehicle with fixed wings.',
    'bus': 'A large motor vehicle designed to carry passengers, usually along a fixed route.',
    'train': 'A series of connected vehicles that run on a track and transport people or cargo.',
    'truck': 'A motor vehicle designed for transporting goods, typically larger than a car.',
    'boat': 'A small to medium-sized watercraft used for travel on water.',
    'traffic light': 'A signaling device positioned at road intersections to control vehicle and pedestrian traffic.',
    'fire hydrant': 'A pipe with a valve and spout where firefighters can connect their hoses to access water.',
    'stop sign': 'A traffic sign instructing drivers to halt and ensure the intersection is clear before proceeding.',
    'parking meter': 'A device used to collect money for parking a vehicle in a specific area.',
    'bench': 'A long seat for multiple people, often found in parks and public places.',
    'bird': 'A warm-blooded, egg-laying animal with feathers, wings, and a beak.',
    'cat': 'A small, domesticated carnivorous mammal with soft fur and a distinctive purring sound.',
    'dog': 'A domesticated mammal known for loyalty, varied breeds, and diverse sizes.',
    'horse': 'A large, hoofed mammal used for riding and farm work.',
    'sheep': 'A woolly, domesticated mammal known for grazing and producing wool.',
    'cow': 'A large domesticated mammal raised for milk and meat.',
    'elephant': 'A large mammal with a trunk, known for its size and intelligence.',
    'bear': 'A large, powerful mammal with thick fur and sharp claws.',
    'zebra': 'An African mammal with black-and-white striped fur.',
    'giraffe': 'A tall, long-necked mammal native to Africa, known for its unique spots.',
    'backpack': 'A bag with shoulder straps, carried on the back and used for storage.',
    'umbrella': 'A collapsible device used to shield from rain or sunlight.',
    'handbag': 'A small bag carried by hand or over the shoulder, often used for personal items.',
    'tie': 'A long, narrow piece of cloth worn around the neck, often with formal clothing.',
    'suitcase': 'A portable case for holding clothes and personal items during travel.',
    'frisbee': 'A plastic disc thrown and caught as a game or sport.',
    'skis': 'Long, flat runners attached to boots for gliding over snow.',
    'snowboard': 'A flat board used to slide down snowy slopes.',
    'sports ball': 'A ball used for various sports, such as soccer, basketball, or football.',
    'kite': 'A light frame with thin material flown in the wind by means of a string.',
    'baseball bat': 'A smooth, wooden or metal club used in baseball to hit the ball.',
    'baseball glove': 'A leather glove worn by a baseball player for catching the ball.',
    'skateboard': 'A short board with small wheels for riding or performing tricks.',
    'surfboard': 'A long, narrow board used for riding waves in the ocean.',
    'tennis racket': 'A handled frame with a netted area, used for hitting the ball in tennis.',
    'bottle': 'A container with a narrow neck, used for holding liquids.',
    'wine glass': 'A glass with a stem used for drinking wine.',
    'cup': 'A small container used for drinking beverages.',
    'fork': 'A utensil with prongs used for eating or serving food.',
    'knife': 'A tool with a sharp blade used for cutting.',
    'spoon': 'A utensil with a shallow bowl for scooping food.',
    'bowl': 'A round, deep dish used for holding food or liquid.',
    'banana': 'A long, curved fruit with yellow skin and soft, sweet flesh.',
    'apple': 'A round fruit with smooth skin and crisp flesh, usually red, green, or yellow.',
    'sandwich': 'Two or more slices of bread with filling between them.',
    'orange': 'A citrus fruit with a tough, bright orange skin and juicy segments inside.',
    'broccoli': 'A green vegetable with a tree-like structure.',
    'carrot': 'An orange root vegetable, typically crunchy and sweet.',
    'hot dog': 'A cooked sausage served in a sliced bun, often with condiments.',
    'pizza': 'A dish made with a round, flat base of dough topped with sauce, cheese, and various toppings.',
    'donut': 'A sweet, deep-fried dough ring or ball, often glazed or topped with sugar.',
    'cake': 'A baked dessert typically made from flour, sugar, and other ingredients.',
    'chair': 'A piece of furniture designed for sitting, typically with four legs and a backrest.',
    'couch': 'A long, upholstered piece of furniture for sitting or reclining.',
    'potted plant': 'A plant grown in a container or pot for decorative purposes.',
    'bed': 'A piece of furniture for sleeping or resting.',
    'dining table': 'A table used for eating meals.',
    'toilet': 'A plumbing fixture for disposing of human waste.',
    'tv': 'An electronic device for receiving television broadcasts.',
    'laptop': 'A portable computer with an integrated screen and keyboard.',
    'mouse': 'A small device used to control the pointer on a computer screen.',
    'remote': 'A handheld device for controlling electronic equipment from a distance.',
    'keyboard': 'A panel of keys used to input data into a computer or typewriter.',
    'cell phone': 'A handheld electronic device for communication.',
    'microwave': 'An appliance that heats and cooks food using electromagnetic radiation.',
    'oven': 'An appliance for baking or roasting food.',
    'toaster': 'An appliance for browning bread slices.',
    'sink': 'A basin with a faucet for washing hands, dishes, etc.',
    'refrigerator': 'An appliance for keeping food and drinks cool.',
    'book': 'A set of printed or written pages bound together, containing text or images.',
    'clock': 'A device that shows the time of day.',
    'vase': 'A decorative container used to hold flowers.',
    'scissors': 'A tool with two blades used for cutting.',
    'teddy bear': 'A soft toy in the shape of a bear, often made of fabric and stuffing.',
    'hair drier': 'An electric device for drying hair by blowing hot air.',
    'toothbrush': 'A small brush used for cleaning teeth.',
    'background': 'The surrounding environment or backdrop in an image that is not the main subject, often consisting of surfaces, objects, or scenery.'
}

def get_desc_embed():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load('ViT-B/32', device)

  labels_desc = [key + ": " + value for key, value in labels_dict.items()] 
  coco_classes = [key for key in labels_dict]

  text_inputs = torch.cat([clip.tokenize(desc) for desc in labels_desc]).to(device)

  with torch.no_grad():
      labels_enc = model.encode_text(text_inputs)

  return labels_enc
