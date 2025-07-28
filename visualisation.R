library(tidyverse)

all_files <- Sys.glob("~/Downloads/jenna_df/*.csv.gz")

track_descriptions <- read_csv("~/Documents/GitHub/borzoi/jenna/track_descriptions.csv", col_names = 'description') %>%
  mutate(track = 1:dplyr::n() - 1) %>%
  mutate(description = str_remove(description, "RNA:"))

rough_start = 187965
rough_end = 361399

read_df <- function(filename){
  df <- read_csv(filename)  %>%
    mutate(pos = 1:dplyr::n()) %>%
    pivot_longer(cols = -pos, names_to = 'track', values_to = 'value') %>%
    # mutate(rough_coordinate = 166175000 - 169000 + 32*pos) %>% wrong...
    filter(pos > rough_start/32 & pos < rough_end/32) %>%
    mutate(track = as.numeric(track)) %>%
    left_join(track_descriptions) %>%
    mutate(filename = word(filename, -1, sep="/"))
}

all_df <- map_df(all_files, read_df)

# ggplot(df, aes(x = pos, y = value, colour = track %in% c(17, 18, 19))) +
#   geom_point(alpha = 0.1) +
#   facet_wrap(~track %in% 17:19)

sums_df <- all_df %>%
  group_by(track, filename) %>%
  mutate(sum = sum(value)) %>%
  ungroup() %>%
  distinct(filename, track, sum, description)

just_brain <- sums_df %>%
  filter(str_detect(description, 'brain')) %>%
  group_by(filename) %>%
  mutate(mean_sum = mean(sum)) %>%
  distinct(filename, mean_sum)

best_filename <- just_brain$filename[which(just_brain$mean_sum == max(just_brain$mean_sum))]
worst_filename <- just_brain$filename[which(just_brain$mean_sum == min(just_brain$mean_sum))]
intron_filename <- just_brain$filename[which(str_detect(just_brain$filename, "166127502C"))]
distal_filename <- just_brain$filename[which(str_detect(just_brain$filename, "166149042C"))]

simple_names_df <- data.frame(filename = c(best_filename, worst_filename, intron_filename, distal_filename),
                              simple_name = factor(c("Proximal Promoter", "Wild-Type", "Proximal Intron 1", "Distal Promoter"),
                                                   levels = c("Wild-Type", "Proximal Intron 1", "Proximal Promoter", "Distal Promoter")))

ggplot(sums_df, aes(x = description, y = sum, fill = track %in% 17:19)) +
  geom_dotplot(binaxis = 'y', stackdir = 'center') +
  ggeasy::easy_rotate_x_labels(angle = -45) +
  facet_wrap(~filename) +
  ggeasy::easy_remove_legend()



sums_df2 <- sums_df %>%
  inner_join(simple_names_df)

ggplot(sums_df2 %>% filter(filename %in% c(best_filename, worst_filename)), aes(x = description, y = sum, fill = track %in% 17:19)) +
  geom_dotplot(binaxis = 'y', stackdir = 'center') +
  ggeasy::easy_rotate_x_labels(angle = -45) +
  facet_wrap(~simple_name) +
  ggeasy::easy_remove_legend()

ggplot(sums_df2, aes(x = description, y = sum, fill = track %in% 17:19)) +
  geom_dotplot(binaxis = 'y', stackdir = 'center') +
  ggeasy::easy_rotate_x_labels(angle = -45) +
  facet_wrap(~simple_name) +
  ggeasy::easy_remove_legend()

ggplot(sums_df2 %>% filter(track %in% 17:19), aes(x = simple_name, y = sum, fill = simple_name)) +
  geom_col() +
  ylim(0, NA) +
  theme_classic() +
  ggeasy::easy_remove_legend() +
  xlab("Position of mutations") +
  ylab("Total SCN1A expression")



# ggplot(all_df %>% filter(track %in% 17:19), aes(x = pos, y = value, colour = str_detect(filename, ';'))) +
#   geom_point(alpha = 0.1) +
#   facet_wrap(~str_detect(filename, ';'))

# ggplot(all_df %>% filter(filename %in% c(best_filename, worst_filename)) %>% 
#          filter(track %in% 17:19), aes(x = pos, y = value)) +
#   geom_area(alpha = 1) +
#   facet_wrap(~filename)

ggplot(all_df %>% inner_join(simple_names_df) %>% filter(filename %in% c(best_filename, worst_filename)) %>% 
         filter(track %in% 17:19), aes(x = pos, y = value)) +
  geom_area(alpha = 1) +
  facet_wrap(~simple_name)

ggplot(all_df %>% inner_join(simple_names_df) %>% filter(filename %in% c(intron_filename, worst_filename)) %>% 
         filter(track %in% 17:19), aes(x = pos, y = value)) +
  geom_area(alpha = 1) +
  facet_wrap(~simple_name)

ggplot(all_df %>% inner_join(simple_names_df) %>% filter(filename %in% c(distal_filename, worst_filename)) %>% 
         filter(track %in% 17:19), aes(x = pos, y = value)) +
  geom_area(alpha = 1) +
  facet_wrap(~simple_name)

