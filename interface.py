import pygame

screen_width=800
screen_height=600

bgcolor="BLACK"
text_color="WHITE"

color_active = pygame.Color('lightskyblue3') 
  
color_passive = pygame.Color('chartreuse4') 
color = color_passive 


pygame.init()

sentence = []

base_font = pygame.font.Font(None, 32) 

window = pygame.display.set_mode((screen_width,screen_height))
pygame.display.set_caption("ABCDEFGH")
input_rect=pygame.Rect(screen_width/2-100, screen_height/2-20,200,40)

def input_box(user_text,color):
    pygame.draw.rect(window,color,input_rect,2)
    text_surface = base_font.render(user_text, True, text_color) 
      
    window.blit(text_surface, (input_rect.x+5, input_rect.y+5))

    input_rect.w = max(100, text_surface.get_width()+10) 


def draw(user_text, color):
    window.fill(bgcolor)
    input_box(user_text, color)

def give_sentence():
    return sentence

def main():
    run=True
    clock=pygame.time.Clock()
    FPS=60
    active=False
    user_text=''
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
            if event.type == pygame.MOUSEBUTTONDOWN: 
                if input_rect.collidepoint(event.pos): 
                    active = True
                else: 
                    active = False
    
            if event.type == pygame.KEYDOWN: 
    
                if event.key == pygame.K_BACKSPACE:  
                    user_text = user_text[:-1] 
                elif event.key == pygame.K_RETURN:
                    sentence.append(user_text)
                    
                    print(sentence)
                else: 
                    user_text += event.unicode

        if active:
            color=color_active
        else:
            color=color_passive
        
        draw(user_text, color)

        # for i in range(len(sentence)):
        #     text_surface = base_font.render(sentence[i], True, text_color) 

        #     window.blit(text_surface, (input_rect.x-100, input_rect.y+100+i*50))


        pygame.display.update()
    pygame.quit()

if __name__=="__main__":
    main()