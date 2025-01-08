def get_page_id(address:int, size:int):
    begin_page = address // 4096
    end_page = (address + size - 1) // 4096
    return begin_page, end_page

def default_hot_function(item,threshold) -> bool:
    return item["access"] / (item["last_access"]-item["enqueue_instr"]) >= threshold

def locate_and_get_int(line:str, sub:str):
    start_index = line.find(sub)
    if start_index != -1:
        line = line.replace('>',' ')
        int_str = line[start_index+len(sub):].split()[0]
        try:
            int_val = int(int_str)
        except ValueError:
            int_val = -1 #failed to convert
        return int_val
    return -1