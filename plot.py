class PlotImages:
    # Inherit this try to use makegrid
    def __init__(
        self,
        dataloader,
        image_matrix_array: np.array = None,
        n_rows: int = 1,
        title_list: Optional[List[str]] = None,
        **kwargs
    ):
        # assume detached: https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
        # TODO: Implement detached
        self.dataloader = dataloader
        self.fig = go.Figure
        self.image_matrix_array = image_matrix_array
        self.title_list = title_list
        self.n_rows = n_rows
        self.image_size = config.image_size
        # self.kwargs = kwargs

    def _denormalize(self, image):
        # takes in an image of C x H x W or batch of images of B x C x H x W
        image = image.numpy().transpose(
            1, 2, 0
        )  # CV2 or PIL images have channels last, but PyTorch is channels first, so need transpose the channels
        mean = [0.485, 0.456, 0.406]
        stdd = [0.229, 0.224, 0.225]
        image = (image * stdd + mean).clip(0, 1)
        return image

    def get_aug_params(self):
        return str(
            getattr(
                AlbumentationsAugmentation(transforms=transforms_train), "transforms"
            )
        )

    def _reset_fig(self):
        # reset figure
        self.fig = go.Figure()

    def _get_next_batch(self):
        # returns a batch of image BxCxHxW of tensors
        image_batch, image_labels = next(iter(self.dataloader))
        return image_batch, image_labels

    # This will be useful so we can construct the corresponding mask
    def _get_img_id(img_path):
        img_basename = os.path.basename(img_path)
        img_id = os.path.splitext(img_basename)[0][: -len("_sat")]
        return img_id

    def _yield_image(self, image_paths: str):
        # https://www.kaggle.com/abhmul/python-image-generator-tutorial
        # https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
        pass

    def plot_multiple_img(self, **kwargs):
        # pull default values dict here, we can update the dict along the way
        subplot_dict = generate_default_configuration(
            lambda item_name: getattr(plotly.subplots, item_name),
            item_name="make_subplots",
        )
        n_cols = math.ceil(len(self.image_matrix_array) / self.n_rows)
        subplot_dict["rows"] = self.n_rows
        subplot_dict["cols"] = n_cols
        # subplot_dict['horizontal_spacing'] = 0.00001#0.2/n_cols
        subplot_dict["vertical_spacing"] = 0.1  # 0.3/self.n_rows
        subplot_dict["subplot_titles"] = (
            self.title_list if self.title_list is not None else 1
        )

        fig = plotly.subplots.make_subplots(**subplot_dict)

        counter = 1
        a = 1
        for index, img in enumerate(self.image_matrix_array):
            # ncols = 4
            # nrows = 3
            # 0//4 = 0
            row_index = counter + index // n_cols
            col_index = index % n_cols + 1
            # let us say we have 12 images, with the desired rows and columns to be rows=3, columns=4
            # then in the first iteration, we want (1,1), (1,2), (1,3), (1,4)
            # The index in enumerate will be: 0,1,2,3 and we use modulo for column to get 1,2,3,4 but
            # rmb to +1 behind since in Plotly we want 1,2,3,4 and not 0,1,2,3
            fig.add_trace(
                go.Image(z=self.image_matrix_array[index], name="a"),
                row=row_index,
                col=col_index,
            )
            # if (index+1)%ncols==0:
            #     counter+=1

        # set height and width to be dynamic,
        # width should be num of images * image size + 100
        # height should be num of images in columns * image size + 100
        layout_height = self.n_rows * self.image_size + 100
        layout_width = n_cols * self.image_size + 100

        fig.update_layout(
            height=layout_height,
            width=layout_width,
            title_text="Transformed Images",
            title_x=0.5,
        )  # title_x for middle alignment

        fig.show()
