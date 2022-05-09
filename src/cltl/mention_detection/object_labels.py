from dataclasses import dataclass
from enum import Enum, auto

class ObjectType(Enum):
    CAT = "cat"
    DOG = "dog"
    PHONE = "phone"
    TV = "tv"
    PERSON = "person"
    LAPTOP = "laptop"
    BOTTLE = "bottle"
    CUP = "cup"
    BOOK = "book"

options = [ObjectType.CAT,
               ObjectType.DOG,
               ObjectType.PHONE,
               ObjectType.TV,
               ObjectType.PERSON,
               ObjectType.LAPTOP,
               ObjectType.BOTTLE,
               ObjectType.CUP,
               ObjectType.BOOK,
               ObjectType.TV]

def main():
    print((ObjectType(2)) )
if __name__ == '__main__':
    main()
